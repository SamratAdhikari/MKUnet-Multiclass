import argparse
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def patient_id(path):
    match = re.search(r"patient(\d{3})", path.name)

    if not match:
        raise ValueError(f"Cannot parse patient ID from {path}")

    return int(match.group(1))


def split_for(pid, train_end=80, val_end=100):
    if pid <= train_end:
        return "train"
    if pid <= val_end:
        return "val"
    return "test"


def normalize_volume(volume):
    volume = volume.astype(np.float32)
    finite = volume[np.isfinite(volume)]

    if finite.size == 0:
        return np.zeros_like(volume, dtype=np.float32)

    low, high = np.percentile(finite, [1, 99])

    if high <= low:
        low, high = finite.min(), finite.max()

    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)

    volume = np.clip(volume, low, high)
    volume = (volume - low) / (high - low)

    return volume.astype(np.float32)


def main(args):
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    pairs = []

    for gt in raw_root.rglob("patient*_frame*_gt.nii"):
        image = gt.with_name(gt.name.replace("_gt.nii", ".nii"))

        if image.exists():
            pairs.append((image, gt))

    pairs.sort(key=lambda p: (patient_id(p[0]), p[0].name))

    if not pairs:
        raise RuntimeError(f"No ACDC image/GT pairs found under {raw_root}")

    shutil.rmtree(out_root, ignore_errors=True)

    counts = {"train": 0, "val": 0, "test": 0}
    patients = {"train": set(), "val": set(), "test": set()}

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
            raise ValueError(f"Unexpected labels in {gt_path}: min={label.min()}, max={label.max()}")

        image = normalize_volume(image)
        stem = image_path.stem

        for z in range(image.shape[2]):
            out_path = split_dir / f"{stem}_z{z:03d}.npz"

            np.savez_compressed(
                out_path,
                image=image[:, :, z],
                label=label[:, :, z].astype(np.uint8)
            )

            counts[split] += 1

        patients[split].add(pid)

    print("\nFinished.")

    for split in ["train", "val", "test"]:
        ids = sorted(patients[split])

        print(
            f"{split}: {len(ids)} patients, "
            f"{counts[split]} slices, "
            f"patient{ids[0]:03d}-patient{ids[-1]:03d}"
        )

    print(f"Image/GT pairs: {len(pairs)}")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--out_root", default="./data/ACDC_npz")
    parser.add_argument("--train_end", type=int, default=80)
    parser.add_argument("--val_end", type=int, default=100)

    main(parser.parse_args())