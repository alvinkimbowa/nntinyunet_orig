import argparse
import random
import numpy as np
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """2x (conv + relu)"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_stages, features_start):
        super(UNet, self).__init__()
        features = features_start
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # Down path
        self.down_layers.append(DoubleConv(in_channels, features))
        feats = [features]
        for _ in range(num_stages - 1):
            self.down_layers.append(DoubleConv(features, features * 2))
            features *= 2
            feats.append(features)

        # Up path
        for i in range(num_stages - 1, 0, -1):
            self.up_layers.append(nn.ConvTranspose2d(features, features // 2, kernel_size=2, stride=2))
            self.up_layers.append(DoubleConv(features, features // 2))
            features //= 2

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        encs = []
        for down in self.down_layers:
            x = down(x)
            encs.append(x)
            x = self.pool(x)
        # bottom
        x = encs.pop()
        for idx in range(0, len(self.up_layers), 2):
            x = self.up_layers[idx](x)
            enc = encs.pop()
            if x.shape != enc.shape:
                diffY = enc.size()[2] - x.size()[2]
                diffX = enc.size()[3] - x.size()[3]
                x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([enc, x], dim=1)
            x = self.up_layers[idx+1](x)
        x = self.final_conv(x)
        return x


def _install_naswot_hooks(model, batch_size):
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

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            if module.inplace:
                module.inplace = False
            module.visited_backwards = False
            handles.append(module.register_forward_hook(forward_hook))
            if hasattr(module, "register_full_backward_hook"):
                handles.append(module.register_full_backward_hook(backward_hook))
            else:
                handles.append(module.register_backward_hook(backward_hook))

    return handles, K_accum


def naswot_score(model, x):
    model.zero_grad(set_to_none=True)
    handles, K = _install_naswot_hooks(model, x.size(0))
    x = x.clone().requires_grad_(True)
    y = model(x)
    y.backward(torch.ones_like(y))
    _ = model(x.detach())
    for h in handles:
        h.remove()
    _, logdet = np.linalg.slogdet(K)
    return float(logdet)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Compute NASWOT score for a UNet width")
    parser.add_argument("--width", type=int, default=16, help="base width")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--num_stages", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    x = torch.randn(args.batch_size, args.in_channels, args.image_size, args.image_size, device=device)

    model = UNet(args.in_channels, args.out_channels, args.num_stages, args.width).to(device)
    scores = []
    for _ in range(args.batches):
        scores.append(naswot_score(model, x))
    avg = float(np.nanmean(scores))
    print(f"width={args.width} naswot={avg}")


if __name__ == "__main__":
    main()
