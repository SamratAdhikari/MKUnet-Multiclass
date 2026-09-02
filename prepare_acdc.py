"""
Prepare the raw ACDC 2017 training set into simple 2D NPZ slices.

Expected raw files somewhere under --raw_root:
  patientXXX_frameYY.nii.gz
  patientXXX_frameYY_gt.nii.gz

The official ACDC training set has labels 0..3:
  0 background, 1 right ventricle, 2 myocardium, 3 left ventricle.

Default patient-level split used by THIS extension package:
  patient001-070 -> train
  patient071-080 -> val
  patient081-100 -> test
This is a reproducible experimental split, not a claim that it is the only
canonical ACDC split.
"""
import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def patient_id(path):
    m = re.search(r"patient(\d{3})", path.name)
    if not m:
        raise ValueError(f"Cannot parse patient id from {path}")
    return int(m.group(1))


def split_for(pid, train_end=70, val_end=80):
    if pid <= train_end:
        return "train"
    if pid <= val_end:
        return "val"
    return "test"


def normalize_volume(vol):
    vol = vol.astype(np.float32)
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo)).astype(np.float32)


def main(args):
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    pairs = []
    for gt in raw_root.rglob("patient*_frame*_gt.nii.gz"):
        image = Path(str(gt).replace("_gt.nii.gz", ".nii.gz"))
        if image.exists():
            pairs.append((image, gt))
    pairs.sort(key=lambda p: (patient_id(p[0]), p[0].name))
    if not pairs:
        raise RuntimeError(f"No ACDC image/GT pairs found under {raw_root}")

    counts = {"train": 0, "val": 0, "test": 0}
    for image_path, gt_path in tqdm(pairs, desc="Preparing ACDC"):
        pid = patient_id(image_path)
        split = split_for(pid, args.train_end, args.val_end)
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        image = np.asarray(nib.load(str(image_path)).get_fdata(), dtype=np.float32)
        label = np.asarray(nib.load(str(gt_path)).get_fdata(), dtype=np.int16)
        if image.shape != label.shape:
            raise ValueError(f"Shape mismatch: {image_path} {image.shape} vs {gt_path} {label.shape}")
        if label.min() < 0 or label.max() > 3:
            raise ValueError(f"Unexpected ACDC labels in {gt_path}: min={label.min()}, max={label.max()}")

        image = normalize_volume(image)
        # NIfTI layout is normally H x W x Z. Save every labeled 2D slice.
        for z in range(image.shape[2]):
            stem = image_path.name.replace(".nii.gz", "")
            out = split_dir / f"{stem}_z{z:03d}.npz"
            np.savez_compressed(out, image=image[:, :, z], label=label[:, :, z].astype(np.uint8))
            counts[split] += 1

    print("Finished.")
    print("Slice counts:", counts)
    print("Output root:", out_root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", required=True, help="Raw ACDC training directory")
    p.add_argument("--out_root", default="./data/ACDC_npz")
    p.add_argument("--train_end", type=int, default=70)
    p.add_argument("--val_end", type=int, default=80)
    main(p.parse_args())
