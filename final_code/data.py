"\"\"\"Data loading utilities for the online adversarial training pipeline.\"\"\""
from __future__ import annotations

import random
from typing import Tuple

import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def tf_train(img_size: int = 224):
    """
    Return the training transforms: random crop, flip, jitter, and normalize.

    `CIFAR10CatDogDM.setup` uses this before the model sees the collected samples.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def tf_eval(img_size: int = 224, normalize: bool = True):
    """
    Return the eval transforms: resize, center crop, and normalize.

    `CIFAR10CatDogDM.setup` uses this for the validation and test loaders.
    """
    ops = [
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


class CIFAR10CatDogDM(L.LightningDataModule):
    """Binary CIFAR-10 cats/dogs subset with ImageNet-style transforms."""

    CAT, DOG = 3, 5

    def __init__(
        self,
        root: str = "./data",
        batch_size: int = 128,
        num_workers: int = 0,
        img_size: int = 224,
        val_ratio: float = 0.10,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.bs = batch_size
        self.nw = int(num_workers)
        self.img = img_size
        self.val_ratio = float(val_ratio)
        self.pin = bool(pin_memory)
        self.train_ds = self.val_ds = self.test_ds = None

    @staticmethod
    def _remap(y: int) -> int:
        """Return 0 for cat and 1 for dog labels so we get a binary task."""
        return 0 if y == CIFAR10CatDogDM.CAT else 1

    def setup(self, stage=None):  # type: ignore[override]
        """
        Load CIFAR-10 cats and dogs, split them, and wrap each split with transforms.
        """
        tr = datasets.CIFAR10(self.root, train=True, download=True)
        te = datasets.CIFAR10(self.root, train=False, download=True)

        def split_one(ds, label, ratio):
            """Split the indices of `ds` that match `label` into train and val parts."""
            idx = [i for i, (_, y) in enumerate(ds) if y == label]
            random.shuffle(idx)
            k = max(1, int(len(idx) * ratio))
            return idx[k:], idx[:k]

        cat_tr, cat_val = split_one(tr, self.CAT, self.val_ratio)
        dog_tr, dog_val = split_one(tr, self.DOG, self.val_ratio)

        class _SubsetRemap(Dataset):
            """Wrap a subset of CIFAR with the given transform."""
            def __init__(self, base, idxs, tf):
                self.base, self.idxs, self.tf = base, idxs, tf

            def __len__(self):
                """Return how many samples are in this subset."""
                return len(self.idxs)

            def __getitem__(self, i):
                """Load one sample and remap its label to binary."""
                x, y = self.base[self.idxs[i]]
                x = self.tf(x)
                return x, CIFAR10CatDogDM._remap(y)

        self.train_ds = _SubsetRemap(tr, cat_tr + dog_tr, tf_train(self.img))
        self.val_ds = _SubsetRemap(tr, cat_val + dog_val, tf_eval(self.img, normalize=True))
        te_idx = [i for i, (_, y) in enumerate(te) if y in (self.CAT, self.DOG)]
        self.test_ds = _SubsetRemap(te, te_idx, tf_eval(self.img, normalize=True))

    def _dl(self, ds, shuffle):
        """Build a DataLoader for the given subset that the Lightning trainer will iterate."""
        args = dict(
            batch_size=self.bs,
            shuffle=shuffle,
            num_workers=self.nw,
            pin_memory=self.pin,
            persistent_workers=False,
        )
        if self.nw > 0:
            args["prefetch_factor"] = 2
        return DataLoader(ds, **args)

    def train_dataloader(self):  # type: ignore[override]
        """Return the training loader used by `LitBinaryClassifier.training_step`."""
        return self._dl(self.train_ds, True)

    def val_dataloader(self):  # type: ignore[override]
        """Return the validation loader used by `trainer.validate` and the phase callback."""
        return self._dl(self.val_ds, False)

    def test_dataloader(self):  # type: ignore[override]
        """Return the test loader used by `trainer.test` once training completes."""
        return self._dl(self.test_ds, False)


class STL10CatDog(Dataset):
    """Binary STL10 cats/dogs subset used for Phase B online adaptation."""

    CAT, DOG = 3, 5

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        img_size: int = 224,
        transform=None,
    ) -> None:
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported STL10 split: {split}")
        self.base = datasets.STL10(root, split=split, download=True)
        self.transform = transform or tf_train(img_size)
        keep = (self.CAT, self.DOG)
        self.idxs = [i for i, (_, y) in enumerate(self.base) if y in keep]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i: int):
        x, y = self.base[self.idxs[i]]
        if self.transform is not None:
            x = self.transform(x)
        label = 0 if y == self.CAT else 1
        return x, label


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "tf_train",
    "tf_eval",
    "CIFAR10CatDogDM",
    "STL10CatDog",
]
