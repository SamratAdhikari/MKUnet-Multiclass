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

Train/val split
----------------
`train_npz` only ships one pool of 2D slices — there's no separate val
folder the way ACDC's `prepare_acdc.py` produces train/val/test folders.
To still get a per-epoch validation signal (val loss, val IoU, per-class
Dice) without leaking, this module splits the *cases* (patients) found in
`train_npz` into a train subset and a val subset, deterministically, based
on `seed`/`val_fraction`. All slices from a given case always land in the
same subset, so validation slices never share a patient with a training
slice. `test_vol_h5` is untouched and stays the fully held-out test set.
"""

import random
import re
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

# Matches e.g. "case0005" out of "case0005_slice012.npz". Falls back to the
# whole stem if a file doesn't follow the usual naming convention, so a
# funny-named file just becomes its own single-slice "case" rather than
# crashing the split.
_CASE_ID_PATTERN = re.compile(r"(case\d+)", re.IGNORECASE)


def _case_id_from_path(path):
    m = _CASE_ID_PATTERN.search(path.stem)
    return m.group(1).lower() if m else path.stem


def _split_case_ids(case_ids, val_fraction=0.15, seed=42):
    """Deterministically split a set of case ids into (train_ids, val_ids)."""
    case_ids = sorted(case_ids)  # sort first so the shuffle is reproducible
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    n_val = max(1, int(round(len(case_ids) * val_fraction))) if len(case_ids) > 1 else 0
    val_ids = set(case_ids[:n_val])
    train_ids = set(case_ids[n_val:])
    return train_ids, val_ids


def _normalize(image):
    """Standard Synapse preprocessing already maps images to ~[0,1].
    Re-clip defensively then map to [-1, 1] to match the ACDC pipeline."""
    image = np.clip(image, 0.0, 1.0)
    return (image - 0.5) / 0.5


class SynapseTrainDataset(Dataset):
    """2D slices, mirroring ACDCSliceDataset, with a `split` argument
    ("train" / "val" / "all") that partitions by case id."""

    def __init__(self, root, img_size=224, augmentation=False,
                 split="train", val_fraction=0.15, seed=42):
        self.dir = Path(root) / TRAIN_DIR_NAME
        all_files = sorted(self.dir.glob("*.npz"))
        if not all_files:
            raise RuntimeError(f"No NPZ slices found in {self.dir}")

        case_ids = sorted({_case_id_from_path(p) for p in all_files})
        train_ids, val_ids = _split_case_ids(case_ids, val_fraction=val_fraction, seed=seed)

        if split == "train":
            keep_ids = train_ids
        elif split == "val":
            keep_ids = val_ids
        elif split == "all":
            keep_ids = train_ids | val_ids
        else:
            raise ValueError(f"Unknown split {split!r}; expected 'train', 'val', or 'all'.")

        self.files = [p for p in all_files if _case_id_from_path(p) in keep_ids]
        if not self.files:
            raise RuntimeError(
                f"No NPZ slices found for split={split!r} in {self.dir} "
                f"(found {len(case_ids)} case(s) total, val_fraction={val_fraction}). "
                "Try a larger val_fraction if you only have a handful of cases."
            )
        self.split = split
        self.case_ids = keep_ids
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
                              split="train", val_fraction=0.15, seed=42,
                              shuffle=True, num_workers=4):
    """`split="train"` (default) gives the training subset of train_npz's
    cases; `split="val"` gives the held-out validation subset (same
    val_fraction/seed => guaranteed disjoint from the train subset);
    `split="all"` ignores the split and uses every case (old behavior)."""
    ds = SynapseTrainDataset(root, img_size=img_size, augmentation=augmentation,
                              split=split, val_fraction=val_fraction, seed=seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=_train_collate,
    )


def get_synapse_val_loader(root, batch_size, img_size=224, val_fraction=0.15,
                            seed=42, num_workers=2):
    """Fast 2D-slice validation loader meant to run every epoch. Augmentation
    is always off and shuffling is off (order doesn't matter for eval, and
    keeping it fixed makes debugging easier). Uses the same case-level split
    as get_synapse_train_loader(seed=seed, val_fraction=val_fraction), so the
    two are guaranteed not to overlap."""
    return get_synapse_train_loader(
        root, batch_size, img_size=img_size, augmentation=False, split="val",
        val_fraction=val_fraction, seed=seed, shuffle=False, num_workers=num_workers,
    )


def get_synapse_volume_loader(root, img_size=224, num_workers=2):
    """Batch size is always 1 volume at a time (volumes vary in slice count).
    This reads test_vol_h5 — the fully held-out test set, untouched by the
    train/val split above."""
    ds = SynapseVolumeDataset(root, img_size=img_size)
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: b[0],  # single sample, unwrap
    )
