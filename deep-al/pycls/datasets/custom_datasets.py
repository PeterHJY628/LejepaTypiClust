import torch
import torchvision
from PIL import Image
import numpy as np
import pycls.datasets.utils as ds_utils

# MedMNIST dataset name -> (medmnist_key, n_classes, n_channels)
MEDMNIST_META = {
    'PATHMNIST':   ('pathmnist',   9,  3),
    'BLOODMNIST':  ('bloodmnist',  8,  3),
    'ORGANAMNIST': ('organamnist', 11, 1),
    'CHESTMNIST':  ('chestmnist',  14, 1),
    'DERMAMNIST':  ('dermamnist',  7,  3),
    'RETINAMNIST': ('retinamnist', 5,  3),
}


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, transform, test_transform, download=True, only_features=False, feature_source="simclr"):
        super(CIFAR10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features
        self.features = ds_utils.load_features("CIFAR10", train=train, normalized=False, source=feature_source)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)


        return img, target


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train, transform, test_transform, download=True, only_features=False, feature_source="simclr"):
        super(CIFAR100, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features
        self.features = ds_utils.load_features("CIFAR100", train=train, normalized=False, source=feature_source)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.only_features:
            img = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    img = self.test_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)

        return img, target


class STL10(torchvision.datasets.STL10):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(STL10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        self.targets = self.labels

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.transpose(1,2,0))

        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, target


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(MNIST, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(SVHN, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target


class MedMNISTDataset(torch.utils.data.Dataset):
    """
    Unified wrapper for MedMNIST datasets (medmnist v3+).

    Supported names: PATHMNIST, BLOODMNIST, ORGANAMNIST, CHESTMNIST, DERMAMNIST, RETINAMNIST.

    - All images are returned as 3-channel RGB PIL images (grayscale originals are repeated).
    - ChestMNIST (multi-label) labels are converted to a single integer: index of the
      first positive label, or 0 if all labels are zero.
    - RetinaMNIST (ordinal) labels are kept as integer class index (0-4).
    """

    def __init__(self, dataset_name: str, root: str, split: str, transform, test_transform,
                 download: bool = True):
        super().__init__()
        assert dataset_name in MEDMNIST_META, \
            f"Unknown MedMNIST dataset '{dataset_name}'. Known: {list(MEDMNIST_META)}"
        assert split in ('train', 'val', 'test'), f"Invalid split '{split}'"

        import medmnist
        key, n_classes, _ = MEDMNIST_META[dataset_name]
        cls = getattr(medmnist, key.upper().replace('MNIST', 'MNIST'))
        # as_rgb=True ensures all images come out as 3-channel PIL images
        self._ds = cls(split=split, transform=None, download=download,
                       as_rgb=True, root=root if root else None)
        self.n_classes = n_classes
        self.dataset_name = dataset_name
        self.transform = transform
        self.test_transform = test_transform
        self.no_aug = False

        # Build a flat targets list for compatibility with AL samplers
        targets = []
        for i in range(len(self._ds)):
            _, lbl = self._ds[i]
            targets.append(self._to_int(lbl))
        self.targets = targets

    @staticmethod
    def _to_int(label: np.ndarray) -> int:
        """Convert medmnist label array to a single integer class index."""
        flat = label.flatten()
        if len(flat) == 1:
            return int(flat[0])
        # Multi-label (ChestMNIST): index of first positive label, or 0 if none
        pos = np.where(flat > 0)[0]
        return int(pos[0]) if len(pos) > 0 else 0

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int):
        img, label = self._ds[index]
        target = self._to_int(label)

        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, target


class ISIC2019(torchvision.datasets.ImageFolder):
    """
    ISIC2019 dataset wrapper based on folder structure:
      <root>/train/<class_name>/*.jpg
      <root>/test/<class_name>/*.jpg

    When only_features=True the dataset returns pre-extracted SimCLR feature
    vectors (shape 512) loaded from
      scan/results/imagenet/pretext/features_seed<seed>.npy   (train)
      scan/results/imagenet/pretext/test_features_seed<seed>.npy (test)
    instead of raw images, which is required by FeaturesNet / LINEAR_FROM_FEATURES.
    """

    def __init__(self, root, split, transform, test_transform,
                 only_features: bool = False, feature_seed: int = 1):
        assert split in ["train", "test"], f"Unsupported split: {split}"
        split_root = f"{root}/{split}"
        super(ISIC2019, self).__init__(split_root, transform=transform)
        self.test_transform = test_transform
        self.no_aug = False
        self.only_features = only_features
        self.features = None
        if only_features:
            self.features = ds_utils.load_features(
                "ISIC2019", seed=feature_seed,
                train=(split == "train"), normalized=False,
            )

    def __getitem__(self, index: int):
        path, target = self.samples[index]

        if self.only_features:
            return self.features[index], target

        img = self.loader(path)

        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, target