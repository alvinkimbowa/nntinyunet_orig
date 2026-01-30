#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def get_dataset_name(dataset_id, nnunet_raw):
    dataset_id = str(dataset_id).zfill(3)
    dataset_name = [
        d for d in os.listdir(nnunet_raw)
        if d.startswith(f"Dataset{dataset_id}") and os.path.isdir(os.path.join(nnunet_raw, d))
    ]
    if len(dataset_name) != 1:
        raise RuntimeError(f"Found {len(dataset_name)} datasets with id {dataset_id}, expected 1")
    return dataset_name[0]


def load_image(nnunet_raw, dataset_name, img_id):
    img_dir_tr = Path(nnunet_raw) / dataset_name / "imagesTr"
    img_dir_ts = Path(nnunet_raw) / dataset_name / "imagesTs"
    paths = []
    for d in (img_dir_tr, img_dir_ts):
        p0 = d / f"{img_id}_0000.png"
        if p0.exists():
            paths = [p0, d / f"{img_id}_0001.png", d / f"{img_id}_0002.png"]
            break
    if not paths:
        raise FileNotFoundError(f"Missing image for {img_id}")
    if all(p.exists() for p in paths):
        ch = [Image.open(p).convert("L") for p in paths]
        return Image.merge("RGB", ch)
    return Image.open(paths[0]).convert("RGB")


def get_embedding_model(device):
    try:
        import torchvision.models as models
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception:
        # fallback to older API if needed
        import torchvision.models as models
        model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


def get_reducer(method):
    if method == "umap":
        try:
            import umap
        except Exception as e:
            raise RuntimeError("umap-learn not installed. Install or use --method tsne") from e
        return umap.UMAP(n_components=2, random_state=0)
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, random_state=0, init="random", learning_rate="auto")


def main():
    parser = argparse.ArgumentParser(description="Embed batch images (top vs bottom correlation)")
    parser.add_argument("--train_dataset_id", type=int, required=True)
    parser.add_argument("--nas_dir", type=str, default="results/nas_metrics")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--bottom_k", type=int, default=5)
    parser.add_argument("--neutral_k", type=int, default=5, help="batches near zero correlation")
    parser.add_argument("--method", type=str, choices=["umap", "tsne"], default="umap")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overlay_thumbs", action="store_true", help="overlay thumbnails on the plot")
    parser.add_argument("--plot_both", action="store_true", help="save both with and without thumbnails")
    parser.add_argument("--thumb_size", type=int, default=28, help="thumbnail size (pixels)")
    parser.add_argument("--max_thumbs", type=int, default=80, help="max thumbnails to overlay total")
    args = parser.parse_args()

    nnunet_raw = os.environ["nnUNet_raw"]
    dataset_name = get_dataset_name(args.train_dataset_id, nnunet_raw)
    dataset_json = Path(nnunet_raw) / dataset_name / "dataset.json"
    if dataset_json.exists():
        with open(dataset_json, "r") as f:
            dj = json.load(f)
        num_channels = len(dj.get("channel_names", {}))
    else:
        num_channels = 1

    corr_path = Path(args.nas_dir) / f"{dataset_name}_batch_jacobian_corr_b{args.batches}.csv"
    batch_path = Path(args.nas_dir) / f"{dataset_name}_batch_jacobian_b{args.batches}.csv"
    if not corr_path.exists() or not batch_path.exists():
        raise SystemExit("Missing batch jacobian files. Run score_net.py --save_batch_jacobian and analyze_batch_jacobian.py")

    corr = pd.read_csv(corr_path).sort_values("spearman", ascending=False)
    top = corr.head(args.top_k)
    bottom = corr.tail(args.bottom_k)
    neutral = corr.iloc[(corr["spearman"].abs()).argsort()].head(args.neutral_k)

    batch_df = pd.read_csv(batch_path)
    selected = pd.concat(
        [
            top.assign(group="top"),
            bottom.assign(group="bottom"),
            neutral.assign(group="neutral"),
        ],
        ignore_index=True,
    )

    # collect unique img ids
    img_rows = []
    for _, r in selected.iterrows():
        ids = str(r["img_ids"]).split(";")
        for img_id in ids:
            img_rows.append({"img_id": img_id, "group": r["group"], "batch": r["batch"], "spearman": r["spearman"]})
    img_df = pd.DataFrame(img_rows).drop_duplicates("img_id")

    device = torch.device("cpu" if args.gpu < 0 else f"cuda:{args.gpu}")
    model = get_embedding_model(device)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    feats = []
    for img_id in img_df["img_id"].tolist():
        img = load_image(nnunet_raw, dataset_name, img_id)
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(x).detach().cpu().numpy().squeeze()
        feats.append(emb)
    feats = np.stack(feats, axis=0)

    reducer = get_reducer(args.method)
    emb2d = reducer.fit_transform(feats)

    if args.out:
        base_out = Path(args.out)
    else:
        base_out = Path(args.nas_dir) / f"{dataset_name}_batch_embed_{args.method}.png"

    def plot_scatter(with_thumbs: bool, out_path: Path):
        fig, ax = plt.subplots(figsize=(5, 4))
        for group, color in [("top", "tab:blue"), ("bottom", "tab:orange"), ("neutral", "tab:green")]:
            mask = img_df["group"] == group
            ax.scatter(emb2d[mask, 0], emb2d[mask, 1], s=18, alpha=0.8, label=group, color=color)
        title = f"{dataset_name} batch embeddings ({args.method})"
        if with_thumbs:
            title += " + thumbs"
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend()

        if with_thumbs:
            thumb_indices = list(range(len(img_df)))
            if len(thumb_indices) > args.max_thumbs:
                thumb_indices = thumb_indices[: args.max_thumbs]
            for idx in thumb_indices:
                img_id = img_df.iloc[idx]["img_id"]
                img = load_image(nnunet_raw, dataset_name, img_id)
                img = img.resize((args.thumb_size, args.thumb_size))
                imagebox = OffsetImage(np.array(img), zoom=1)
                ab = AnnotationBbox(imagebox, (emb2d[idx, 0], emb2d[idx, 1]), frameon=False)
                ax.add_artist(ab)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Wrote {out_path}")

    if args.plot_both:
        plot_scatter(False, base_out)
        plot_scatter(True, base_out.with_name(base_out.stem + "_thumbs.png"))
    else:
        plot_scatter(args.overlay_thumbs, base_out)

    out_csv = base_out.with_suffix(".csv")
    out_df = img_df.copy()
    out_df["x"] = emb2d[:, 0]
    out_df["y"] = emb2d[:, 1]
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
