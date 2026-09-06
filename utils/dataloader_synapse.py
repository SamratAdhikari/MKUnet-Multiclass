"""
Dataloader for the Synapse multi-organ CT dataset.

Assumes the standard preprocessing layout used by TransUNet / Swin-UNet /
HiFormer / MISSFormer and most public mirrors (including the common Kaggle
copies of this dataset):

    <root>/
        train_npz/
            case0005_slice000.npz   # keys: 'image' (H,W float), 'label' (H,W uint8)
            case0005_slice001.npz
            ...
        test_vol_h5/
            case0001.npy.h5         # keys: 'image' (D,H,W float), 'label' (D,H,W uint8)
            case0002.npy.h5
            ...

Images are already windowed/clipped and rescaled to roughly [0, 1] by the
standard preprocessing script. If your unpacked Kaggle copy uses different
folder/file names, just adjust TRAIN_DIR_NAME / TEST_DIR_NAME below (or pass
different `root` subfolders) — everything else stays the same.

9 segmentation classes (standard Synapse ordering):
    0 background, 1 aorta, 2 gallbladder, 3 kidney(L), 4 kidney(R),
    5 liver, 6 pancreas, 7 spleen, 8 stomach
"""

from pathlib import Path

import albumentations as A
import h5py
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

TRAIN_DIR_NAME = "train_npz"
TEST_DIR_NAME = "test_vol_h5"

NUM_CLASSES = 9
CLASS_NAMES = [
    "background", "aorta", "gallbladder", "kidney_L", "kidney_R",
    "liver", "pancreas", "spleen", "stomach",
]


def _normalize(image):
    """Standard Synapse preprocessing already maps images to ~[0,1].
    Re-clip defensively then map to [-1, 1] to match the ACDC pipeline."""
    image = np.clip(image, 0.0, 1.0)
    return (image - 0.5) / 0.5


class SynapseTrainDataset(Dataset):
    """2D slices for training, mirroring ACDCSliceDataset."""

    def __init__(self, root, img_size=224, augmentation=False):
        self.dir = Path(root) / TRAIN_DIR_NAME
        self.files = sorted(self.dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No NPZ slices found in {self.dir}")
        self.img_size = int(img_size)

        transforms = []
        if augmentation:
            transforms += [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
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

        out = self.transform(image=image, mask=label)
        image_t = out["image"].float()
        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
        image_t = _normalize(image_t)
        label_t = out["mask"].long()
        if label_t.ndim == 3 and label_t.shape[0] == 1:
            label_t = label_t.squeeze(0)
        return image_t, label_t


class SynapseVolumeDataset(Dataset):
    """Whole 3D volumes for evaluation (Dice is usually reported per-case,
    reconstructed from per-slice predictions), mirroring the TransUNet-style
    test protocol. Returns everything needed to run slice-wise inference and
    re-stack the volume."""

    def __init__(self, root, img_size=224):
        self.dir = Path(root) / TEST_DIR_NAME
        self.files = sorted(self.dir.glob("*.h5")) + sorted(self.dir.glob("*.npy.h5"))
        self.files = sorted(set(self.files))
        if not self.files:
            raise RuntimeError(f"No H5 volumes found in {self.dir}")
        self.img_size = int(img_size)
        self.resize = A.Resize(self.img_size, self.img_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with h5py.File(path, "r") as f:
            image = f["image"][:].astype(np.float32)  # (D, H, W)
            label = f["label"][:].astype(np.uint8)     # (D, H, W)

        orig_h, orig_w = image.shape[1], image.shape[2]
        slices = np.stack(
            [self.resize(image=image[i])["image"] for i in range(image.shape[0])],
            axis=0,
        )
        slices = _normalize(slices.astype(np.float32))
        image_t = torch.from_numpy(slices).unsqueeze(1).float()  # (D, 1, h, w)
        label_t = torch.from_numpy(label.astype(np.int64))       # (D, H, W) original size
        return image_t, label_t, (orig_h, orig_w), path.stem


def _train_collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    return images, labels


def get_synapse_train_loader(root, batch_size, img_size=224, augmentation=False,
                              shuffle=True, num_workers=4):
    ds = SynapseTrainDataset(root, img_size=img_size, augmentation=augmentation)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=_train_collate,
    )


def get_synapse_volume_loader(root, img_size=224, num_workers=2):
    """Batch size is always 1 volume at a time (volumes vary in slice count)."""
    ds = SynapseVolumeDataset(root, img_size=img_size)
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: b[0],  # single sample, unwrap
    )
