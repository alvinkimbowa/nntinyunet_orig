import torch


def _get_output_tensor(y):
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y


def _scalar_loss_from_output(y):
    y = _get_output_tensor(y)
    return y.float().mean()


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


def jacobian_score(model, x):
    model.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    y = model(x)
    y = _get_output_tensor(y)
    loss = y.float().sum()
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
