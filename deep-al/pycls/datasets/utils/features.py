import numpy as np
import torch
DATASET_FEATURES_DICT = {
    'train':
        {
            'ISIC2019': '../../scan/results/isic2019/pretext/features_seed{seed}.npy',
            'CIFAR10':'../../scan/results/cifar-10/pretext/features_seed{seed}.npy',
            'CIFAR100':'../../scan/results/cifar-100/pretext/features_seed{seed}.npy',
            'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/features_seed{seed}.npy',
            'IMAGENET50': '../../dino/runs/trainfeat.pth',
            'IMAGENET100': '../../dino/runs/trainfeat.pth',
            'IMAGENET200': '../../dino/runs/trainfeat.pth',
            'PATHMNIST':   '../../scan/results/pathmnist/pretext/features_seed{seed}.npy',
            'BLOODMNIST':  '../../scan/results/bloodmnist/pretext/features_seed{seed}.npy',
            'ORGANAMNIST': '../../scan/results/organamnist/pretext/features_seed{seed}.npy',
            'CHESTMNIST':  '../../scan/results/chestmnist/pretext/features_seed{seed}.npy',
            'DERMAMNIST':  '../../scan/results/dermamnist/pretext/features_seed{seed}.npy',
            'RETINAMNIST': '../../scan/results/retinamnist/pretext/features_seed{seed}.npy',
        },
    'test':
        {
            'ISIC2019': '../../scan/results/isic2019/pretext/test_features_seed{seed}.npy',
            'CIFAR10': '../../scan/results/cifar-10/pretext/test_features_seed{seed}.npy',
            'CIFAR100': '../../scan/results/cifar-100/pretext/test_features_seed{seed}.npy',
            'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/test_features_seed{seed}.npy',
            'IMAGENET50': '../../dino/runs/testfeat.pth',
            'IMAGENET100': '../../dino/runs/testfeat.pth',
            'IMAGENET200': '../../dino/runs/testfeat.pth',
            'PATHMNIST':   '../../scan/results/pathmnist/pretext/test_features_seed{seed}.npy',
            'BLOODMNIST':  '../../scan/results/bloodmnist/pretext/test_features_seed{seed}.npy',
            'ORGANAMNIST': '../../scan/results/organamnist/pretext/test_features_seed{seed}.npy',
            'CHESTMNIST':  '../../scan/results/chestmnist/pretext/test_features_seed{seed}.npy',
            'DERMAMNIST':  '../../scan/results/dermamnist/pretext/test_features_seed{seed}.npy',
            'RETINAMNIST': '../../scan/results/retinamnist/pretext/test_features_seed{seed}.npy',
        }
}

def load_features(ds_name, seed=1, train=True, normalized=True, source='simclr'):
    " load pretrained features for a dataset "
    if source == 'lejepa':
        split = "train" if train else "test"
        fname_dict = {
            "train": {
                "CIFAR10": "../../scan/results/cifar-10/lejepa/features_seed{seed}.npy",
                "CIFAR100": "../../scan/results/cifar-100/lejepa/features_seed{seed}.npy",
                "TINYIMAGENET": "../../scan/results/tiny-imagenet/lejepa/features_seed{seed}.npy",
            },
            "test": {
                "CIFAR10": "../../scan/results/cifar-10/lejepa/test_features_seed{seed}.npy",
                "CIFAR100": "../../scan/results/cifar-100/lejepa/test_features_seed{seed}.npy",
                "TINYIMAGENET": "../../scan/results/tiny-imagenet/lejepa/test_features_seed{seed}.npy",
            },
        }
        fname = fname_dict[split][ds_name].format(seed=seed)
    elif source == 'dinov2':
        split = "train" if train else "test"
        fname_dict = {
            "train": {
                "CIFAR10": "../../scan/results/cifar-10/dinov2/features_seed{seed}.npy",
            },
            "test": {
                "CIFAR10": "../../scan/results/cifar-10/dinov2/test_features_seed{seed}.npy",
            },
        }
        fname = fname_dict[split][ds_name].format(seed=seed)
    else:
        split = "train" if train else "test"
        fname = DATASET_FEATURES_DICT[split][ds_name].format(seed=seed)

    if fname.endswith('.npy'):
        features = np.load(fname)
    elif fname.endswith('.pth'):
        features = torch.load(fname)
    else:
        raise Exception("Unsupported filetype")
    if normalized:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features