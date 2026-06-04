#!/usr/bin/env python3
"""
Train a LeJEPA-style ResNet-18 encoder on CIFAR-10 and export features.

Default output directory is separated from SimCLR to avoid overwriting:
  - results/cifar-10/lejepa/features_seed{seed}.npy
  - results/cifar-10/lejepa/test_features_seed{seed}.npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18


class MultiViewCIFAR10(Dataset):
    def __init__(self, root: str, train: bool, n_views: int, download: bool = True):
        self.n_views = n_views
        self.base = datasets.CIFAR10(root=root, train=train, download=download)
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010],
                    ),
                ]
            )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        views = [self.transform(img) for _ in range(self.n_views)]
        return torch.stack(views, dim=0), label


class ResNet18LeJEPA(nn.Module):
    def __init__(self, proj_dim: int = 128, feat_dim: int = 512):
        super().__init__()
        backbone = resnet18(num_classes=1000)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(in_features, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        # x: [N, V, C, H, W]
        n, v = x.shape[:2]
        flat = x.flatten(0, 1)
        emb = self.backbone(flat)                     # [N*V, 512]
        proj = self.projector(emb)                    # [N*V, proj_dim]
        proj = proj.reshape(n, v, -1).transpose(0, 1) # [V, N, proj_dim]
        return emb, proj


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        # proj: [V, N, D]
        a = torch.randn(proj.size(-1), 256, device=proj.device)
        a = a / (a.norm(p=2, dim=0, keepdim=True) + 1e-12)
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats = []
    for views, _ in loader:
        x = views[:, 0].to(device, non_blocking=True)  # single deterministic view
        emb = model.backbone(x)
        feats.append(emb.detach().cpu().numpy())
    arr = np.concatenate(feats, axis=0)
    return arr


def main():
    parser = argparse.ArgumentParser(description="LeJEPA CIFAR-10 pretraining for TypiClust RP")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lambda_sigreg", type=float, default=0.05)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Linear warmup epochs before cosine decay.",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=1e-3,
        help="Final LR is lr * min_lr_ratio.",
    )
    parser.add_argument(
        "--feature_subdir",
        type=str,
        default="lejepa",
        help="Subdir under results/cifar-10 for exported features (e.g. pretext, lejepa).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this LeJEPA training script.")
    device = torch.device("cuda")

    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "datasets" / "cifar-10"
    out_dir = root_dir / "results" / "cifar-10" / args.feature_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MultiViewCIFAR10(str(data_dir), train=True, n_views=args.views, download=True)
    test_ds = MultiViewCIFAR10(str(data_dir), train=False, n_views=1, download=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_eval_loader = DataLoader(
        MultiViewCIFAR10(str(data_dir), train=True, n_views=1, download=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    net = ResNet18LeJEPA(proj_dim=args.proj_dim).to(device)
    sigreg = SIGReg().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = min(max(args.warmup_epochs * steps_per_epoch, 1), total_steps - 1) if total_steps > 1 else 1
    warmup = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        opt,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=args.lr * args.min_lr_ratio,
    )
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])

    for epoch in range(args.epochs):
        net.train()
        running = 0.0
        steps = 0
        for views, _ in train_loader:
            views = views.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, proj = net(views)
                inv_loss = (proj.mean(0, keepdim=False) - proj).square().mean()
                sig_loss = sigreg(proj)
                loss = args.lambda_sigreg * sig_loss + (1.0 - args.lambda_sigreg) * inv_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()

            running += float(loss.item())
            steps += 1

        avg_loss = running / max(steps, 1)
        current_lr = opt.param_groups[0]["lr"]
        print(f"[Seed {args.seed}] Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.6f} - lr: {current_lr:.6e}")

    ckpt = out_dir / f"lejepa_resnet18_seed{args.seed}.pth"
    torch.save({"model": net.state_dict(), "seed": args.seed, "epochs": args.epochs}, ckpt)

    train_feats = extract_features(net, train_eval_loader, device)
    test_feats = extract_features(net, test_loader, device)

    np.save(out_dir / f"features_seed{args.seed}.npy", train_feats)
    np.save(out_dir / f"test_features_seed{args.seed}.npy", test_feats)
    print(f"Saved train features: {out_dir / f'features_seed{args.seed}.npy'}")
    print(f"Saved test features: {out_dir / f'test_features_seed{args.seed}.npy'}")
    print(f"Saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
