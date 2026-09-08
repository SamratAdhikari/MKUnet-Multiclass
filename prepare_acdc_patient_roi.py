import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

from mkunet_network import MK_UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_RE = re.compile(r"patient(?P<pid>\d{3})_frame(?P<frame>\d+)_z(?P<z>\d+)\.npz$")


def parse_name(path):
    match = FILE_RE.search(path.name)
    if not match:
        raise ValueError(
            f"Unexpected ACDC NPZ filename: {path.name}. "
            "Expected patientXXX_frameYY_zZZZ.npz"
        )
    return int(match.group("pid")), int(match.group("frame")), int(match.group("z"))


def load_localizer(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "ROI localizer checkpoint must be produced by train_acdc_roi_localizer.py"
        )

    model = MK_UNet(
        num_classes=int(checkpoint.get("num_classes", 2)),
        in_channels=int(checkpoint.get("in_channels", 1)),
        channels=checkpoint["channels"],
        deep_supervision=bool(checkpoint.get("deep_supervision", True)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, int(checkpoint.get("img_size", 224)), checkpoint


def largest_component(mask):
    """Keep the largest binary component to suppress distant false positives."""
    if not mask.any():
        return mask

    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask

    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return labeled == keep


@torch.no_grad()
def predict_whole_heart(model, image, model_size, threshold):
    """Predict whole-heart mask and return it at the NPZ slice's native size."""
    h, w = image.shape
    x = torch.from_numpy(image.astype(np.float32))[None, None].to(device)
    x = F.interpolate(x, size=(model_size, model_size), mode="bilinear", align_corners=False)

    logits = model(x)[0]
    probs = torch.softmax(logits, dim=1)[:, 1:2]
    probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)

    mask = probs[0, 0].cpu().numpy() >= threshold
    return largest_component(mask)


def robust_patient_union(slice_masks, h, w, max_center_distance_ratio=0.30):
    """Combine predicted cardiac masks from all ED/ES slices for one patient.

    A single ROI is shared by every slice from the patient. Obvious spatial
    outlier predictions are rejected before the union is computed.
    """
    valid = []
    centers = []

    for mask in slice_masks:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        centers.append((float(ys.mean()), float(xs.mean())))
        valid.append(mask)

    if not valid:
        return np.zeros((h, w), dtype=bool), 0, 0

    centers_arr = np.asarray(centers, dtype=np.float32)
    median_center = np.median(centers_arr, axis=0)
    distances = np.sqrt(((centers_arr - median_center) ** 2).sum(axis=1))
    max_distance = max_center_distance_ratio * min(h, w)

    accepted = [m for m, d in zip(valid, distances) if d <= max_distance]
    if not accepted:
        accepted = valid

    union = np.zeros((h, w), dtype=bool)
    for mask in accepted:
        union |= mask

    return union, len(valid), len(accepted)


def square_roi_from_union(union, h, w, margin_ratio=0.25, min_side=160):
    """Create a square whole-heart ROI from the patient-level prediction union."""
    ys, xs = np.nonzero(union)
    if len(xs) == 0:
        side = min(max(min_side, int(round(0.70 * min(h, w)))), max(h, w))
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        return square_box_from_center(cy, cx, side)

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    box_h = y1 - y0
    box_w = x1 - x0
    side = int(np.ceil(max(box_h, box_w) * (1.0 + 2.0 * margin_ratio)))
    side = max(side, min_side)

    cy = (y0 + y1 - 1) / 2.0
    cx = (x0 + x1 - 1) / 2.0
    return square_box_from_center(cy, cx, side)


def square_box_from_center(cy, cx, side):
    """Return possibly out-of-image square coordinates (y0, y1, x0, x1)."""
    side = int(max(1, side))
    y0 = int(round(cy - side / 2.0))
    x0 = int(round(cx - side / 2.0))
    return y0, y0 + side, x0, x0 + side


def crop_with_padding(array, box, fill_value=0):
    """Crop a 2D array. Pad if the square ROI extends beyond image boundaries."""
    y0, y1, x0, x1 = box
    side_h = y1 - y0
    side_w = x1 - x0
    if side_h != side_w:
        raise ValueError("ROI box must be square")

    out = np.full((side_h, side_w), fill_value, dtype=array.dtype)

    src_y0 = max(0, y0)
    src_y1 = min(array.shape[0], y1)
    src_x0 = max(0, x0)
    src_x1 = min(array.shape[1], x1)

    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out

    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return out


def group_files(data_root):
    """Group all files by split -> patient, combining both ED/ES frames."""
    grouped = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}

    for split in grouped:
        split_dir = data_root / split
        files = sorted(split_dir.glob("*.npz"))
        if not files:
            raise RuntimeError(f"No NPZ files found in {split_dir}")

        for path in files:
            pid, frame, z = parse_name(path)
            grouped[split][pid].append((frame, z, path))

        for pid in grouped[split]:
            grouped[split][pid].sort(key=lambda item: (item[0], item[1]))

    return grouped


def prepare_patient(model, model_size, entries, out_dir, threshold, margin_ratio, min_side):
    # Every slice for a patient should share the same in-plane matrix size after
    # your existing 1.25 mm resampling. We verify it explicitly.
    cached = []
    predicted_masks = []
    expected_shape = None

    for frame, z, path in entries:
        with np.load(path) as d:
            image = d["image"].astype(np.float32)
            label = d["label"].astype(np.uint8)
            spacing = d["spacing"].astype(np.float32)

        if image.shape != label.shape:
            raise ValueError(f"Shape mismatch in {path}: {image.shape} vs {label.shape}")

        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise ValueError(
                f"Patient has inconsistent in-plane shapes after preprocessing: "
                f"expected {expected_shape}, found {image.shape} in {path}"
            )

        pred_mask = predict_whole_heart(model, image, model_size, threshold)
        predicted_masks.append(pred_mask)
        cached.append((frame, z, path, image, label, spacing))

    h, w = expected_shape
    union, n_valid, n_accepted = robust_patient_union(predicted_masks, h, w)
    box = square_roi_from_union(
        union,
        h,
        w,
        margin_ratio=margin_ratio,
        min_side=min_side,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    for frame, z, path, image, label, spacing in cached:
        image_roi = crop_with_padding(image, box, fill_value=0.0).astype(np.float32)
        label_roi = crop_with_padding(label, box, fill_value=0).astype(np.uint8)

        np.savez_compressed(
            out_dir / path.name,
            image=image_roi,
            label=label_roi,
            spacing=spacing,
            roi_box=np.asarray(box, dtype=np.int32),
            original_shape=np.asarray([h, w], dtype=np.int32),
        )

    return {
        "roi_box_y0_y1_x0_x1": [int(v) for v in box],
        "original_shape": [int(h), int(w)],
        "roi_shape": [int(box[1] - box[0]), int(box[3] - box[2])],
        "slices_with_localizer_prediction": int(n_valid),
        "predictions_kept_after_outlier_filter": int(n_accepted),
        "used_center_fallback": bool(not union.any()),
    }


def main(args):
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    checkpoint_path = Path(args.localizer_checkpoint)

    if data_root.resolve() == out_root.resolve():
        raise ValueError("--out_root must be different from --data_root")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Localizer checkpoint not found: {checkpoint_path}")

    model, model_size, checkpoint = load_localizer(checkpoint_path)
    grouped = group_files(data_root)

    shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_data_root": str(data_root),
        "localizer_checkpoint": str(checkpoint_path),
        "localizer_network": checkpoint.get("network"),
        "localizer_img_size": model_size,
        "threshold": args.threshold,
        "margin_ratio": args.margin_ratio,
        "min_side": args.min_side,
        "note": (
            "ROI coordinates are determined only from localizer predictions. "
            "Ground-truth masks are cropped after the ROI is fixed and are never "
            "used to choose validation/test ROI coordinates."
        ),
        "patients": {},
    }

    print("Device:", device)
    print("Source full-slice dataset:", data_root)
    print("Output patient-ROI dataset:", out_root)
    print("Localizer:", checkpoint_path)

    total_files = 0

    for split in ["train", "val", "test"]:
        metadata["patients"][split] = {}
        split_out = out_root / split

        for pid, entries in tqdm(
            sorted(grouped[split].items()),
            desc=f"Creating {split} patient ROIs",
        ):
            patient_meta = prepare_patient(
                model,
                model_size,
                entries,
                split_out,
                threshold=args.threshold,
                margin_ratio=args.margin_ratio,
                min_side=args.min_side,
            )
            metadata["patients"][split][f"patient{pid:03d}"] = patient_meta
            total_files += len(entries)

    with open(out_root / "roi_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nFinished patient-level ROI preparation.")
    print(f"Saved {total_files} cropped slices")
    print(f"Metadata: {out_root / 'roi_metadata.json'}")
    print("\nTrain your existing segmentation code with:")
    print(f"  python train_acdc.py --data_root {out_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        default="./data/ACDC_npz",
        help="Existing full-slice NPZ dataset produced by prepare_acdc.py.",
    )
    p.add_argument("--out_root", default="./data/ACDC_patient_roi")
    p.add_argument(
        "--localizer_checkpoint",
        default="./model_pth_acdc_roi_localizer/roi_localizer-best.pth",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Whole-heart localizer foreground probability threshold.",
    )
    p.add_argument(
        "--margin_ratio",
        type=float,
        default=0.25,
        help="Extra margin on each side of the predicted whole-heart extent.",
    )
    p.add_argument(
        "--min_side",
        type=int,
        default=160,
        help="Minimum square ROI side in native resampled pixels. At 1.25 mm spacing, 160 px = 200 mm.",
    )
    main(p.parse_args())
