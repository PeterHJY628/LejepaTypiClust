#!/usr/bin/env python3
"""
Train a light DINOv2-style student/teacher model on CIFAR-10 and export embeddings.

Outputs (default):
  results/cifar-10/dinov2/features_seed{seed}.npy
  results/cifar-10/dinov2/test_features_seed{seed}.npy
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18


class DINOViewCIFAR10(Dataset):
    def __init__(self, root: str, train: bool, download: bool = True):
        self.base = datasets.CIFAR10(root=root, train=train, download=download)
        self.train = train
        self.global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(24, scale=(0.2, 0.6)),
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                ),
            ]
        )
        self.eval_transform = transforms.Compose(
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
        if not self.train:
            return self.eval_transform(img), label
        views = [
            self.global_transform(img),
            self.global_transform(img),
            self.local_transform(img),
            self.local_transform(img),
        ]
        return views, label


class DINOHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, bottleneck_dim),
        )
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last.weight_g.data.fill_(1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last(x)


class DINOModel(nn.Module):
    def __init__(self, proj_dim: int = 2048):
        super().__init__()
        backbone = resnet18(num_classes=1000)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = DINOHead(feat_dim, out_dim=proj_dim)

    def forward(self, x):
        z = self.backbone(x)
        p = self.head(z)
        return z, p


def dino_loss(student_out, teacher_out, t_student=0.1, t_teacher=0.04):
    s = F.log_softmax(student_out / t_student, dim=-1)
    t = F.softmax(teacher_out / t_teacher, dim=-1).detach()
    return -(t * s).sum(dim=-1).mean()


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    for s, t in zip(student.parameters(), teacher.parameters()):
        t.data.mul_(momentum).add_(s.data, alpha=1.0 - momentum)


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        z, _ = model(x)
        feats.append(z.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def main():
    parser = argparse.ArgumentParser(description="DINOv2-style CIFAR10 pretraining")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--teacher_momentum", type=float, default=0.996)
    parser.add_argument("--proj_dim", type=int, default=2048)
    parser.add_argument("--feature_subdir", type=str, default="dinov2")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DINOv2 training.")
    device = torch.device("cuda")

    root = Path(__file__).resolve().parent
    data_dir = root / "datasets" / "cifar-10"
    out_dir = root / "results" / "cifar-10" / args.feature_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DINOViewCIFAR10(str(data_dir), train=True, download=True)
    train_eval_ds = DINOViewCIFAR10(str(data_dir), train=False, download=True)
    test_ds = DINOViewCIFAR10(str(data_dir), train=False, download=True)
    test_ds.base = datasets.CIFAR10(root=str(data_dir), train=False, download=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
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

    student = DINOModel(proj_dim=args.proj_dim).to(device)
    teacher = DINOModel(proj_dim=args.proj_dim).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)

    for epoch in range(args.epochs):
        student.train()
        running = 0.0
        n_steps = 0
        for views, _ in train_loader:
            global1 = views[0].to(device, non_blocking=True)
            global2 = views[1].to(device, non_blocking=True)
            local1 = views[2].to(device, non_blocking=True)
            local2 = views[3].to(device, non_blocking=True)

            _, s_g1 = student(global1)
            _, s_g2 = student(global2)
            _, s_l1 = student(local1)
            _, s_l2 = student(local2)

            with torch.no_grad():
                _, t_g1 = teacher(global1)
                _, t_g2 = teacher(global2)

            loss = (
                dino_loss(s_g1, t_g2)
                + dino_loss(s_g2, t_g1)
                + dino_loss(s_l1, t_g1)
                + dino_loss(s_l2, t_g2)
            ) / 4.0

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            cur_m = 1.0 - (1.0 - args.teacher_momentum) * (
                math.cos(math.pi * epoch / max(args.epochs - 1, 1)) + 1.0
            ) / 2.0
            update_teacher(student, teacher, cur_m)

            running += float(loss.item())
            n_steps += 1

        sched.step()
        avg_loss = running / max(n_steps, 1)
        print(
            f"[Seed {args.seed}] Epoch {epoch + 1}/{args.epochs} "
            f"loss={avg_loss:.6f} lr={opt.param_groups[0]['lr']:.6e}"
        )

    train_feats = extract_features(teacher, train_eval_loader, device)
    test_feats = extract_features(teacher, test_loader, device)

    np.save(out_dir / f"features_seed{args.seed}.npy", train_feats)
    np.save(out_dir / f"test_features_seed{args.seed}.npy", test_feats)
    torch.save({"teacher": teacher.state_dict(), "seed": args.seed}, out_dir / f"dinov2_resnet18_seed{args.seed}.pth")

    print(f"Saved train features: {out_dir / f'features_seed{args.seed}.npy'}")
    print(f"Saved test features : {out_dir / f'test_features_seed{args.seed}.npy'}")


if __name__ == "__main__":
    main()
