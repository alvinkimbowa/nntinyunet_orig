import os
import numpy as np
import torch
import re

import matplotlib.pyplot as plt


def _get_output_tensor(y):
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y


def _scalar_loss_from_output(y):
    y = _get_output_tensor(y)
    return y.float().mean()


def _get_stage_prefixes(model):
    enc_idxs = []
    for name, _ in model.named_modules():
        m = re.search(r"encoder\.stages\.(\d+)", name)
        if m:
            enc_idxs.append(int(m.group(1)))
    if not enc_idxs:
        return None, None
    return min(enc_idxs), max(enc_idxs)


def _install_naswot_hooks(model, batch_size, stage_only=False):
    handles = []
    K_accum = np.zeros((batch_size, batch_size), dtype=np.float32)

    def forward_hook(module, inp, _):
        try:
            if not getattr(module, "visited_backwards", False):
                return
            x = inp[0]
            x = x.view(x.size(0), -1)
            x = (x > 0).float()
            K = x @ x.t()
            K2 = (1.0 - x) @ (1.0 - x.t())
            K_accum[:] = K_accum + K.cpu().numpy() + K2.cpu().numpy()
        except Exception:
            pass

    def backward_hook(module, *_):
        module.visited_backwards = True

    enc_first = enc_last = None
    if stage_only:
        enc_first, enc_last = _get_stage_prefixes(model)

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            if stage_only:
                if enc_first is None:
                    continue
                keep = (
                    name.startswith(f"encoder.stages.{enc_first}") or
                    name.startswith(f"encoder.stages.{enc_last}")
                )
                if not keep:
                    continue
            if module.inplace:
                module.inplace = False
            module.visited_backwards = False
            handles.append(module.register_forward_hook(forward_hook))
            if hasattr(module, "register_full_backward_hook"):
                handles.append(module.register_full_backward_hook(backward_hook))
            else:
                handles.append(module.register_backward_hook(backward_hook))

    return handles, K_accum


def naswot_score(model, x, stage_only=False):
    model.zero_grad(set_to_none=True)
    handles, K = _install_naswot_hooks(model, x.size(0), stage_only=stage_only)
    x = x.clone().requires_grad_(True)
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y.backward(torch.ones_like(y))
    _ = model(x.detach())
    for h in handles:
        h.remove()
    _, logdet = np.linalg.slogdet(K)
    return float(logdet)


def naswot_module_contributions(model, x):
    model.zero_grad(set_to_none=True)
    batch_size = x.size(0)
    handles = []
    K_by_module = {}

    def forward_hook(module, inp, _):
        try:
            if not getattr(module, "visited_backwards", False):
                return
            key = getattr(module, "_naswot_key", None)
            if key is None:
                return
            xh = inp[0]
            xh = xh.view(xh.size(0), -1)
            xh = (xh > 0).float()
            K = xh @ xh.t()
            K2 = (1.0 - xh) @ (1.0 - xh.t())
            K_by_module[key] += K.cpu().numpy() + K2.cpu().numpy()
        except Exception:
            pass

    def backward_hook(module, *_):
        module.visited_backwards = True

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            if module.inplace:
                module.inplace = False
            module.visited_backwards = False
            module._naswot_key = name
            K_by_module[name] = np.zeros((batch_size, batch_size), dtype=np.float32)
            handles.append(module.register_forward_hook(forward_hook))
            if hasattr(module, "register_full_backward_hook"):
                handles.append(module.register_full_backward_hook(backward_hook))
            else:
                handles.append(module.register_backward_hook(backward_hook))

    x = x.clone().requires_grad_(True)
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y.backward(torch.ones_like(y))
    _ = model(x.detach())
    for h in handles:
        h.remove()

    scores = []
    for name, K in K_by_module.items():
        _, logdet = np.linalg.slogdet(K)
        scores.append((name, float(logdet)))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


def save_activation_distributions(model, x, out_dir, bins=100, max_samples=200000, save_input_hist=True):
    os.makedirs(out_dir, exist_ok=True)
    handles = []
    stats = {}

    def _sample(arr):
        if arr.size <= max_samples:
            return arr
        idx = np.random.choice(arr.size, size=max_samples, replace=False)
        return arr[idx]

    def forward_hook(module, inp, out):
        try:
            name = getattr(module, "_act_debug_name", None)
            if name is None:
                return
            inp_t = inp[0].detach().float()
            out_t = _get_output_tensor(out).detach().float()
            inp_arr = inp_t.reshape(-1).cpu().numpy()
            out_arr = out_t.reshape(-1).cpu().numpy()
            stats[name] = (_sample(inp_arr), _sample(out_arr))
        except Exception:
            pass

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.Conv2d, torch.nn.InstanceNorm2d)):
            if getattr(module, "inplace", False):
                module.inplace = False
            module._act_debug_name = name
            handles.append(module.register_forward_hook(forward_hook))

    if save_input_hist:
        x_arr = x.detach().float().reshape(-1).cpu().numpy()
        x_arr = _sample(x_arr)
        x_mean = float(np.mean(x_arr))
        x_std = float(np.std(x_arr))
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.hist(x_arr, bins=bins, color="tab:green", alpha=0.8)
        ax.set_title("model input")
        ax.text(
            0.98,
            0.95,
            f"mean={x_mean:.4g}\nstd={x_std:.4g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "model_input.png"), dpi=150)
        plt.close(fig)

    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    for name, (inp_arr, out_arr) in stats.items():
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].hist(inp_arr, bins=bins, color="tab:blue", alpha=0.8)
        axes[0].set_title("input")
        inp_mean = float(np.mean(inp_arr))
        inp_std = float(np.std(inp_arr))
        axes[0].text(
            0.98,
            0.95,
            f"mean={inp_mean:.4g}\nstd={inp_std:.4g}",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )
        axes[1].hist(out_arr, bins=bins, color="tab:orange", alpha=0.8)
        axes[1].set_title("activation")
        out_mean = float(np.mean(out_arr))
        out_std = float(np.std(out_arr))
        axes[1].text(
            0.98,
            0.95,
            f"mean={out_mean:.4g}\nstd={out_std:.4g}",
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )
        fig.suptitle(name)
        fig.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        fig.savefig(os.path.join(out_dir, f"{safe_name}.png"), dpi=150)
        plt.close(fig)


def _prefix_up_to(name: str, token: str) -> str | None:
    parts = name.split(".")
    if token not in parts:
        return None
    idx = parts.index(token)
    if idx + 1 >= len(parts):
        return None
    return ".".join(parts[: idx + 2])


def aggregate_naswot_contributions(scores, level="stage"):
    if level not in ("stage", "block", "convblock"):
        raise ValueError(f"Unknown level: {level}")
    agg = {}
    if level == "stage":
        token = "stages"
    elif level == "convblock":
        token = "convs"
    else:
        token = "blocks"
    for name, val in scores:
        key = _prefix_up_to(name, token)
        if key is None:
            continue
        agg[key] = agg.get(key, 0.0) + float(val)
    return sorted(agg.items(), key=lambda t: t[1], reverse=True)


def synflow_score(model, input_shape, device):
    model.zero_grad(set_to_none=True)
    signs = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                signs[name] = torch.sign(p.data)
                p.data = p.data.abs()
    x = torch.ones(input_shape, device=device)
    y = model(x)
    loss = _scalar_loss_from_output(y)
    loss.backward()
    score = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                score += torch.sum(torch.abs(p.grad * p)).item()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data * signs[name]
    return float(score)


def gradnorm_score(model, x):
    model.zero_grad(set_to_none=True)
    y = model(x)
    loss = _scalar_loss_from_output(y)
    loss.backward()
    score = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                score += p.grad.norm(2).item()
    return float(score)


def snip_score(model, x):
    model.zero_grad(set_to_none=True)
    y = model(x)
    loss = _scalar_loss_from_output(y)
    loss.backward()
    score = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                score += torch.sum(torch.abs(p.grad * p)).item()
    return float(score)


def jacobian_score(model, x, y=None, loss_fn=None):
    model.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    y = model(x)
    y = _get_output_tensor(y)
    loss = y.float().sum()
    # loss = y.float().mean()
    # loss = (y ** 2).mean()
    # loss = loss_fn(y, y)
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
    return float(grad.norm().item())


def fisher_score(model, x):
    model.zero_grad(set_to_none=True)
    y = model(x)
    loss = _scalar_loss_from_output(y)
    loss.backward()
    score = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                score += torch.sum(p.grad * p.grad).item()
    return float(score)
