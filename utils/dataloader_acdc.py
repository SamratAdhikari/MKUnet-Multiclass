from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class ACDCSliceDataset(Dataset):
    """2D ACDC NPZ slices created by prepare_acdc.py."""

    def __init__(self, root, split="train", img_size=224, augmentation=False):
        self.files = sorted((Path(root) / split).glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No NPZ files found in {Path(root) / split}")
        self.split = split
        self.img_size = int(img_size)
        transforms = []
        if split == "train" and augmentation:
            transforms += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
            ]
        transforms += [
            A.Resize(self.img_size, self.img_size),
            ToTensorV2(),
        ]
        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path) as d:
            image = d["image"].astype(np.float32)
            label = d["label"].astype(np.uint8)

        h, w = label.shape
        out = self.transform(image=image, mask=label)
        image_t = out["image"].float()
        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
        # prepare_acdc stores [0,1]; map roughly to [-1,1].
        image_t = (image_t - 0.5) / 0.5
        label_t = out["mask"].long()
        if label_t.ndim == 3 and label_t.shape[0] == 1:
            label_t = label_t.squeeze(0)

        if self.split == "train":
            return image_t, label_t
        return image_t, label_t, (h, w), path.stem


def get_acdc_loader(root, split, batch_size, img_size=224, augmentation=False,
                    shuffle=None, num_workers=4):
    ds = ACDCSliceDataset(root, split=split, img_size=img_size, augmentation=augmentation)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
