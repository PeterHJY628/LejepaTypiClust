#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare embedding quality: SimCLR vs LeJEPA."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Evaluate on train or test embeddings",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=10000,
        help="Max samples used for metrics (for speed).",
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=20,
        help="k for kNN label-consistency metric.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="CIFAR10 root path (default: scan/datasets/cifar-10)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: scan/results/cifar-10/embedding_metrics)",
    )
    return parser.parse_args()


def load_embedding(scan_root: Path, family: str, split: str, seed: int):
    stem = "features" if split == "train" else "test_features"
    p = scan_root / "results" / "cifar-10" / family / f"{stem}_seed{seed}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing embedding file: {p}")
    return np.load(p), p


def knn_label_consistency(x: np.ndarray, y: np.ndarray, k: int):
    # Exclude self neighbor by querying k+1 and dropping index 0.
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(x)
    inds = nbrs.kneighbors(x, return_distance=False)[:, 1:]
    neigh_labels = y[inds]
    return float((neigh_labels == y[:, None]).mean())


def compute_metrics(x: np.ndarray, y: np.ndarray, k: int):
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    km = KMeans(n_clusters=10, random_state=0, n_init=10)
    pred = km.fit_predict(x_norm)

    out = {
        "knn_label_consistency": knn_label_consistency(x_norm, y, k),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "ari": float(adjusted_rand_score(y, pred)),
        "silhouette": float(silhouette_score(x_norm, y, metric="euclidean")),
        "calinski_harabasz": float(calinski_harabasz_score(x_norm, y)),
        "davies_bouldin": float(davies_bouldin_score(x_norm, y)),
    }
    return out


def main():
    args = parse_args()
    scan_root = Path(__file__).resolve().parent

    simclr, simclr_path = load_embedding(scan_root, "pretext", args.split, args.seed)
    lejepa, lejepa_path = load_embedding(scan_root, "lejepa", args.split, args.seed)

    n = min(len(simclr), len(lejepa))
    max_points = min(args.max_points, n)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=max_points, replace=False) if max_points < n else np.arange(n)

    simclr = simclr[idx]
    lejepa = lejepa[idx]

    cifar_root = Path(args.data_root) if args.data_root else (scan_root / "datasets" / "cifar-10")
    ds = datasets.CIFAR10(root=str(cifar_root), train=(args.split == "train"), download=False)
    y = np.asarray(ds.targets)[idx]

    sim_m = compute_metrics(simclr, y, args.knn_k)
    lej_m = compute_metrics(lejepa, y, args.knn_k)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else scan_root / "results" / "cifar-10" / "embedding_metrics"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"seed{args.seed}_{args.split}_simclr_vs_lejepa_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "simclr", "lejepa", "higher_is_better"])
        metric_order = [
            ("knn_label_consistency", True),
            ("nmi", True),
            ("ari", True),
            ("silhouette", True),
            ("calinski_harabasz", True),
            ("davies_bouldin", False),
        ]
        for m, hib in metric_order:
            writer.writerow([m, sim_m[m], lej_m[m], hib])

    # Plot bar chart with raw values.
    metric_names = ["knn_label_consistency", "nmi", "ari", "silhouette", "calinski_harabasz", "davies_bouldin"]
    x = np.arange(len(metric_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, [sim_m[m] for m in metric_names], width, label="SimCLR")
    ax.bar(x + width / 2, [lej_m[m] for m in metric_names], width, label="LeJEPA")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20, ha="right")
    ax.set_title(f"Embedding Metrics (seed={args.seed}, split={args.split})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / f"seed{args.seed}_{args.split}_simclr_vs_lejepa_metrics.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    print(f"SimCLR embedding: {simclr_path}")
    print(f"LeJEPA embedding: {lejepa_path}")
    print(f"Samples used: {max_points}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print("\nSummary:")
    for k in metric_names:
        print(f"  {k}: simclr={sim_m[k]:.6f}, lejepa={lej_m[k]:.6f}")


if __name__ == "__main__":
    main()
