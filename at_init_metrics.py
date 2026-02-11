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

def _install_swap_hooks(model, batch_size):
    handles = []
    # store unique neuron-wise activation patterns across the batch
    seen = set()

    def forward_hook(module, inp, _out):
        try:
            x = inp[0]
            if x.size(0) != batch_size:
                return  # SWAP upper bound logic assumes fixed batch size

            x = x.view(x.size(0), -1)          # [B, N]
            x = (x > 0).to(torch.uint8)        # uint8 for packing
            xt = x.t().contiguous()            # [N, B] neuron-wise patterns

            # pack each length-B bitvector into bytes so we can hash it
            # packbits works on CPU numpy, so move once per hook
            arr = xt.cpu().numpy()             # uint8 {0,1}, shape [N,B]
            packed = np.packbits(arr, axis=1)  # shape [N, ceil(B/8)]

            for row in packed:
                seen.add(row.tobytes())
        except Exception:
            pass

    # attach to ReLU modules (common) or whatever you used for NASWOT
    for m in model.modules():
        if isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            handles.append(m.register_forward_hook(forward_hook))

    def score():
        return len(seen)

    return handles, score

def apply_sam(x, alpha):
    if alpha <= 0:
        return x
    mask = torch.bernoulli(
        torch.full_like(x, 1.0 - alpha)
    )
    return x * mask

def install_ncd_swap_hooks(model, batch_size, alpha=0.0):
    seen = set()
    handles = []

    def hook(module, inp, _out):
        x = inp[0]
        if x.size(0) != batch_size:
            return

        # flatten + SAM
        x = x.view(batch_size, -1)
        x = apply_sam(x, alpha)

        # binarize
        x = (x > 0).to(torch.uint8)

        # neuron-wise patterns
        xt = x.t().contiguous()   # [N, B]

        # pack bits for hashing
        packed = np.packbits(
            xt.cpu().numpy(), axis=1
        )

        for row in packed:
            seen.add(row.tobytes())

    for m in model.modules():
        if isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            handles.append(m.register_forward_hook(hook))

    def score():
        return len(seen)

    return handles, score


def install_ncd_naswot_hooks(model, batch_size, alpha=0.0):
    K_accum = torch.zeros(batch_size, batch_size)
    handles = []

    def hook(module, inp, _out):
        x = inp[0]
        if x.size(0) != batch_size:
            return

        # flatten + SAM
        x = x.view(batch_size, -1)
        x = apply_sam(x, alpha)

        # binarize
        x = (x > 0).float()

        # NASWOT kernel
        K1 = x @ x.t()
        K2 = (1 - x) @ (1 - x.t())
        K_accum.add_((K1 + K2).cpu())

    for m in model.modules():
        if isinstance(m, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            handles.append(m.register_forward_hook(hook))

    def score():
        # log |K|
        return torch.logdet(K_accum + 1e-6 * torch.eye(batch_size))

    return handles, score


def ncd_swap_score(model, x, alpha=0.95):
    import copy
    model = copy.deepcopy(model)
    swap_bn_to_ln(model)
    handles, score_fn = install_ncd_swap_hooks(model, x.size(0), alpha=alpha)
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    return float(score_fn())


def ncd_naswot_score(model, x, alpha=0.95):
    import copy
    model = copy.deepcopy(model)
    swap_bn_to_ln(model)
    handles, score_fn = install_ncd_naswot_hooks(model, x.size(0), alpha=alpha)
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    score = score_fn()
    if torch.is_tensor(score):
        score = score.item()
    return float(score)

def swap_bn_to_ln(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.BatchNorm2d):
            ln = torch.nn.LayerNorm(
                module.num_features,
                elementwise_affine=True
            )
            setattr(model, name, ln)
        else:
            swap_bn_to_ln(module)



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


def swap_score(model, x):
    handles, score_fn = _install_swap_hooks(model, x.size(0))
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    return float(score_fn())


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


def az_nas_score(model, x, offload_to_cpu=True):
    model.zero_grad(set_to_none=True)
    orig_device = next(model.parameters()).device
    if offload_to_cpu and orig_device.type == "cuda":
        model = model.to("cpu")
        x = x.detach().cpu()
    conv_modules = []
    for _name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
            conv_modules.append(module)

    if len(conv_modules) < 2:
        return float("nan")

    # Collect conv features for expressivity/progressivity on CPU to save GPU memory.
    features_cpu = []
    handles = []

    def forward_hook(_module, _inp, out):
        out_t = _get_output_tensor(out)
        if out_t is None:
            return
        if out_t.dim() >= 4:
            features_cpu.append(out_t.detach().cpu())

    for module in conv_modules:
        handles.append(module.register_forward_hook(forward_hook))

    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    if len(features_cpu) < 2:
        return float("nan")

    expressivity_scores = []
    for feat in features_cpu:
        c = feat.size(1)
        feat = feat.permute(0, *range(2, feat.dim()), 1).contiguous().view(-1, c)
        m = feat.mean(dim=0, keepdim=True)
        feat = feat - m
        sigma = torch.mm(feat.t(), feat) / (feat.size(0))
        s = torch.linalg.eigvalsh(sigma)
        s_sum = s.sum()
        if not torch.isfinite(s_sum) or s_sum.item() == 0:
            continue
        prob_s = s / (s_sum + 1e-8)
        score = (-prob_s) * torch.log(prob_s + 1e-8)
        score_val = score.sum().item()
        if np.isfinite(score_val):
            expressivity_scores.append(score_val)
    expressivity_scores = np.array(expressivity_scores)
    if expressivity_scores.size < 2:
        return float("nan")
    progressivity = np.min(expressivity_scores[1:] - expressivity_scores[:-1])
    expressivity = np.sum(expressivity_scores)

    # Compute trainability per pair with a fresh forward to avoid keeping all features on GPU.
    scores = []
    for i in reversed(range(1, len(conv_modules))):
        f_in = None
        f_out = None
        handles = []

        def hook_in(_module, _inp, out):
            nonlocal f_in
            out_t = _get_output_tensor(out)
            if out_t is not None and out_t.dim() >= 4:
                f_in = out_t

        def hook_out(_module, _inp, out):
            nonlocal f_out
            out_t = _get_output_tensor(out)
            if out_t is not None and out_t.dim() >= 4:
                f_out = out_t

        handles.append(conv_modules[i - 1].register_forward_hook(hook_in))
        handles.append(conv_modules[i].register_forward_hook(hook_out))

        _ = model(x)

        for h in handles:
            h.remove()

        if f_in is None or f_out is None:
            continue

        if f_out.grad is not None:
            f_out.grad.zero_()
        if f_in.grad is not None:
            f_in.grad.zero_()
        g_out = torch.ones_like(f_out) * 0.5
        g_out = (torch.bernoulli(g_out) - 0.5) * 2
        g_in = torch.autograd.grad(outputs=f_out, inputs=f_in, grad_outputs=g_out, retain_graph=False)[0]
        if g_out.size() == g_in.size() and torch.all(g_in == g_out):
            continue
        else:
            if g_out.dim() >= 4:
                if g_out.size(2) != g_in.size(2) or g_out.size(3) != g_in.size(3):
                    if g_out.dim() == 4:
                        g_in = torch.nn.functional.adaptive_avg_pool2d(
                            g_in, (g_out.size(2), g_out.size(3))
                        )
                    elif g_out.dim() == 5:
                        g_in = torch.nn.functional.adaptive_avg_pool3d(
                            g_in, (g_out.size(2), g_out.size(3), g_out.size(4))
                        )
            g_out = g_out.permute(0, 2, 3, 1).contiguous().view(-1, g_out.size(1))
            g_in = g_in.permute(0, 2, 3, 1).contiguous().view(-1, g_in.size(1))
            mat = torch.mm(g_in.t(), g_out) / (g_out.size(0))
            if mat.size(0) < mat.size(1):
                mat = mat.t()
            s = torch.linalg.svdvals(mat)
            s_max = s.max().item()
            if not np.isfinite(s_max) or s_max <= 0:
                continue
            scores.append(-s_max - 1 / (s_max + 1e-6) + 2)
    if len(scores) == 0:
        return float("nan")
    trainability = np.mean(scores)

    az_score = float(expressivity + progressivity + trainability)
    if np.isnan(az_score):
        az_score = float("nan")

    if offload_to_cpu and orig_device.type == "cuda":
        model = model.to(orig_device)

    return az_score


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


def jacobian_score(model, x, targets=None, loss_fn=None):
    model.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    y = model(x)
    y = _get_output_tensor(y)
    loss = y.float().sum()
    # loss = y.float().mean()
    # loss = (y ** 2).mean()
    # loss = loss_fn(y, targets)
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
