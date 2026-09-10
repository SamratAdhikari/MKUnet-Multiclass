from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Allow running as: python diagnosis/extract_mkunet_features.py from the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mkunet_network import MK_UNet

try:
    from .common import (
        CLASS_IDS,
        compute_cardiac_features,
        dice_for_class,
        discover_patient_dirs,
        find_frame_path,
        load_nifti,
        metadata_for_patient,
        patient_id_from_name,
        diagnosis_split_for,
        load_diagnosis_split,
    )
except ImportError:
    from common import (
        CLASS_IDS,
        compute_cardiac_features,
        dice_for_class,
        discover_patient_dirs,
        find_frame_path,
        load_nifti,
        metadata_for_patient,
        patient_id_from_name,
        diagnosis_split_for,
        load_diagnosis_split,
    )


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Match prepare_acdc.py: clip 1st/99th percentile, map to [0, 1]."""
    volume = volume.astype(np.float32)
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    volume = np.clip(volume, low, high)
    return ((volume - low) / (high - low)).astype(np.float32)


def build_model(network: str, device: torch.device) -> MK_UNet:
    configs = {
        "MK_UNet_T": [4, 8, 16, 24, 32],
        "MK_UNet_S": [8, 16, 32, 48, 80],
        "MK_UNet": [16, 32, 64, 96, 160],
        "MK_UNet_M": [32, 64, 128, 192, 320],
        "MK_UNet_L": [64, 128, 256, 384, 512],
    }
    if network not in configs:
        raise ValueError(f"Unknown network: {network}")
    return MK_UNet(
        num_classes=4,
        in_channels=1,
        channels=configs[network],
        deep_supervision=True,
    ).to(device)


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    obj = torch.load(str(checkpoint), map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint does not contain a state_dict-like mapping")
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in obj.items()}
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    image_volume: np.ndarray,
    img_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict a 3D mask slice-by-slice using the exact training normalization."""
    x = normalize_volume(image_volume)
    h, w, z = x.shape
    # [Z, 1, H, W]
    slices = torch.from_numpy(np.moveaxis(x, 2, 0)[:, None, :, :]).float()
    preds = []
    for start in range(0, z, batch_size):
        batch = slices[start:start + batch_size].to(device)
        batch = F.interpolate(batch, size=(img_size, img_size), mode="bilinear", align_corners=False)
        batch = (batch - 0.5) / 0.5
        logits = model(batch)[0]
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy().astype(np.uint8)
        preds.append(pred)
    pred_zhw = np.concatenate(preds, axis=0)
    return np.moveaxis(pred_zhw, 0, 2)  # [H, W, Z]


def save_mask(mask: np.ndarray, reference_nii: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference_nii.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference_nii.affine, header), str(path))


def main(opt):
    data_root = Path(opt.data_root)
    checkpoint = Path(opt.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    split_map = load_diagnosis_split(Path(opt.split_csv))

    device = torch.device("cuda" if torch.cuda.is_available() and not opt.cpu else "cpu")
    print("Device:", device)
    model = build_model(opt.network, device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    patient_dirs = discover_patient_dirs(data_root)
    if not patient_dirs:
        raise RuntimeError(f"No patientXXX directories found under {data_root}")

    rows = []
    failures = []
    masks_root = Path(opt.save_masks_dir) if opt.save_masks_dir else None

    for patient_dir in tqdm(patient_dirs, desc="MK-UNet features"):
        pid = patient_id_from_name(patient_dir.name)
        if pid > opt.max_patient:
            continue
        try:
            meta = metadata_for_patient(patient_dir)
            split = diagnosis_split_for(pid, meta["label"], split_map)
            ed_img_path = find_frame_path(patient_dir, pid, meta["ed_frame"], gt=False)
            es_img_path = find_frame_path(patient_dir, pid, meta["es_frame"], gt=False)

            ed_img, ed_zooms, ed_nii = load_nifti(ed_img_path, dtype=np.float32)
            es_img, es_zooms, es_nii = load_nifti(es_img_path, dtype=np.float32)
            ed_pred = predict_volume(model, ed_img, opt.img_size, opt.batch_size, device)
            es_pred = predict_volume(model, es_img, opt.img_size, opt.batch_size, device)

            feats = compute_cardiac_features(
                ed_pred, es_pred, ed_zooms, es_zooms,
                float(meta["height_cm"]), float(meta["weight_kg"]),
            )
            row = {
                "patient_id": pid,
                "patient": f"patient{pid:03d}",
                "split": split,
                "label": meta["label"],
                "ed_frame": meta["ed_frame"],
                "es_frame": meta["es_frame"],
                "feature_source": "mkunet_prediction",
                **feats,
            }

            if opt.with_gt_metrics:
                try:
                    ed_gt_path = find_frame_path(patient_dir, pid, meta["ed_frame"], gt=True)
                    es_gt_path = find_frame_path(patient_dir, pid, meta["es_frame"], gt=True)
                    ed_gt, _, _ = load_nifti(ed_gt_path, dtype=np.uint8)
                    es_gt, _, _ = load_nifti(es_gt_path, dtype=np.uint8)
                    # Concatenate ED and ES in z for one patient-level number per class.
                    pred_pair = np.concatenate([ed_pred, es_pred], axis=2)
                    gt_pair = np.concatenate([ed_gt, es_gt], axis=2)
                    row.update({
                        "seg_dice_rv": dice_for_class(pred_pair, gt_pair, CLASS_IDS["RV"]),
                        "seg_dice_myo": dice_for_class(pred_pair, gt_pair, CLASS_IDS["MYO"]),
                        "seg_dice_lv": dice_for_class(pred_pair, gt_pair, CLASS_IDS["LV"]),
                    })
                except Exception as metric_exc:
                    row.update({"seg_dice_rv": np.nan, "seg_dice_myo": np.nan, "seg_dice_lv": np.nan})
                    print(f"Warning patient{pid:03d}: GT metric failed: {metric_exc}")

            rows.append(row)

            if masks_root is not None:
                save_mask(ed_pred, ed_nii, masks_root / f"patient{pid:03d}_ED_pred.nii.gz")
                save_mask(es_pred, es_nii, masks_root / f"patient{pid:03d}_ES_pred.nii.gz")

        except Exception as exc:
            failures.append((pid, str(exc)))
            if opt.strict:
                raise

    df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    out = Path(opt.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} patient rows to {out}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    if opt.with_gt_metrics and {"seg_dice_rv", "seg_dice_myo", "seg_dice_lv"}.issubset(df.columns):
        print("\nMean patient-level segmentation Dice:")
        print(df[["seg_dice_rv", "seg_dice_myo", "seg_dice_lv"]].mean())
    if failures:
        print(f"\nSkipped {len(failures)} patients:")
        for pid, msg in failures[:20]:
            print(f"  patient{pid:03d}: {msg}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract ACDC diagnosis features from MK-UNet predicted masks.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--network", default="MK_UNet", choices=["MK_UNet_T", "MK_UNet_S", "MK_UNet", "MK_UNet_M", "MK_UNet_L"])
    p.add_argument("--output", default="./diagnosis/results/mkunet_features.csv")
    p.add_argument("--split_csv", required=True, help="CSV created once by create_diagnosis_split.py")
    p.add_argument("--save_masks_dir", default="")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_patient", type=int, default=150)
    p.add_argument("--with_gt_metrics", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--strict", action="store_true")
    main(p.parse_args())
