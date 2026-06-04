#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize SimCLR vs LeJEPA embeddings for one seed."
    )
    parser.add_argument("--seed", type=int, required=True, help="Seed id to visualize")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which embedding split to visualize",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["pca", "tsne", "both"],
        help="2D projection method (or both)",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=5000,
        help="Max number of points to plot per method",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="CIFAR10 root directory (default: scan/datasets/cifar-10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default auto-generated under scan/results/cifar-10/embedding_viz)",
    )
    return parser.parse_args()


def load_embedding(scan_root: Path, family: str, split: str, seed: int):
    stem = "features" if split == "train" else "test_features"
    emb_path = scan_root / "results" / "cifar-10" / family / f"{stem}_seed{seed}.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    return np.load(emb_path), emb_path


def project_to_2d(x: np.ndarray, method: str):
    if method == "pca":
        return PCA(n_components=2, random_state=0).fit_transform(x)
    return TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=0,
    ).fit_transform(x)


def main():
    args = parse_args()
    scan_root = Path(__file__).resolve().parent

    simclr_emb, simclr_path = load_embedding(scan_root, "pretext", args.split, args.seed)
    lejepa_emb, lejepa_path = load_embedding(scan_root, "lejepa", args.split, args.seed)

    n1, n2 = len(simclr_emb), len(lejepa_emb)
    if n1 != n2:
        raise ValueError(f"Embedding length mismatch: simclr={n1}, lejepa={n2}")

    n = n1
    max_points = min(args.max_points, n)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=max_points, replace=False) if max_points < n else np.arange(n)

    simclr_sub = simclr_emb[idx]
    lejepa_sub = lejepa_emb[idx]

    cifar_root = Path(args.data_root) if args.data_root else (scan_root / "datasets" / "cifar-10")
    ds = datasets.CIFAR10(root=str(cifar_root), train=(args.split == "train"), download=False)
    labels = np.array(ds.targets)[idx]

    methods = ["pca", "tsne"] if args.method == "both" else [args.method]
    out_dir = scan_root / "results" / "cifar-10" / "embedding_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for method in methods:
        simclr_2d = project_to_2d(simclr_sub, method)
        lejepa_2d = project_to_2d(lejepa_sub, method)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        cmap = plt.cm.get_cmap("tab10", 10)

        for ax, pts, title in [
            (axes[0], simclr_2d, "SimCLR"),
            (axes[1], lejepa_2d, "LeJEPA"),
        ]:
            sc = ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=labels,
                s=6,
                cmap=cmap,
                alpha=0.7,
                linewidths=0,
            )
            ax.set_title(f"{title} ({method.upper()}, {args.split}, seed={args.seed})")
            ax.set_xticks([])
            ax.set_yticks([])

        cbar = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02)
        cbar.set_label("CIFAR10 class")

        if args.output and len(methods) == 1:
            out_path = Path(args.output)
        elif args.output and len(methods) > 1:
            base = Path(args.output)
            out_path = base.with_name(f"{base.stem}_{method}{base.suffix or '.png'}")
        else:
            out_path = out_dir / f"{method}_{args.split}_seed{args.seed}_simclr_vs_lejepa.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        saved_paths.append(out_path)

    for p in saved_paths:
        print(f"Saved figure: {p}")
    print(f"SimCLR embedding: {simclr_path}")
    print(f"LeJEPA embedding: {lejepa_path}")
    print(f"Points plotted: {len(idx)}")


if __name__ == "__main__":
    main()
