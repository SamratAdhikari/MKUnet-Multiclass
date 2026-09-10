"""
Stage 1 -> Stage 2 bridge for the two-stage diagnosis pipeline.

For every patient in the metadata CSV produced by parse_acdc_metadata.py,
this script:
  1. Loads the full-resolution ED and ES 3D volumes (not the pre-sliced
     npz training data, so we keep true voxel spacing and slice count).
  2. Runs the trained MK-UNet segmentation model slice-by-slice and
     reassembles a 3D prediction volume at the *original* resolution.
  3. Computes clinical biomarkers from the predicted masks:
       - LV / RV end-diastolic and end-systolic volume (EDV, ESV, mL)
       - LV / RV ejection fraction (EF, %)
       - Myocardial mass at ED and ES (g, using 1.05 g/mL tissue density)
       - Mean myocardial wall thickness at ED (mm, distance-transform estimate)
       - Body surface area (BSA, DuBois formula) and BSA-indexed volumes/mass,
         when height/weight are available
  4. Writes one row per patient (features + diagnosis label + split) to a
     biomarkers CSV, which train_diagnosis_classifier.py consumes directly.

These are the same family of handcrafted features used in the original
ACDC challenge paper (Bernard et al., 2018) to classify NOR/MINF/DCM/HCM/RV
from segmentation output with a simple classifier.

Usage:
    python extract_biomarkers.py \
        --metadata_csv ./data/acdc_patient_metadata.csv \
        --run_id ACDC_MK_UNet_bs16_sz224_lr0.0001_e150_seed42_120000 \
        --model_root ./model_pth_acdc \
        --network MK_UNet \
        --img_size 224 \
        --out_csv ./data/acdc_biomarkers.csv
"""
import argparse
import csv
import math

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from mkunet_network import MK_UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_BG, CLASS_RV, CLASS_MYO, CLASS_LV = 0, 1, 2, 3
MYOCARDIAL_DENSITY_G_PER_ML = 1.05

CONFIGS = {
    "MK_UNet_T": [4, 8, 16, 24, 32], "MK_UNet_S": [8, 16, 32, 48, 80],
    "MK_UNet": [16, 32, 64, 96, 160], "MK_UNet_M": [32, 64, 128, 192, 320],
    "MK_UNet_L": [64, 128, 256, 384, 512],
}


def normalize_volume(volume):
    """Identical 1st/99th percentile normalization to prepare_acdc.py, so
    inference sees the same intensity distribution the model was trained on."""
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


def load_model(network, run_id, model_root):
    model = MK_UNet(num_classes=4, in_channels=1, channels=CONFIGS[network], deep_supervision=True).to(device)
    ckpt = f"{model_root}/{run_id}/{run_id}-best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
    model.eval()
    return model


@torch.no_grad()
def segment_volume(model, volume_2d_norm, img_size):
    """volume_2d_norm: (H, W, Z) float32 in [0,1]. Returns (H, W, Z) uint8 labels."""
    h, w, z = volume_2d_norm.shape
    preds = np.zeros((h, w, z), dtype=np.uint8)
    for zi in range(z):
        sl = volume_2d_norm[:, :, zi]
        sl_resized = cv2.resize(sl, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(sl_resized).float().unsqueeze(0).unsqueeze(0)  # (1,1,S,S)
        x = (x - 0.5) / 0.5  # match dataloader_acdc.py normalization
        x = x.to(device)

        logits = model(x)[0]
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, (h, w), mode="bilinear", align_corners=False)
        pred = torch.softmax(logits, dim=1).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        preds[:, :, zi] = pred
    return preds


def voxel_volume_ml(sx, sy, sz):
    return (sx * sy * sz) / 1000.0  # mm^3 -> mL


def class_volume_ml(mask3d, class_id, vox_ml):
    return float((mask3d == class_id).sum()) * vox_ml


def mean_wall_thickness_mm(mask3d, class_id_myo, sx, sy):
    """Rough per-slice wall thickness estimate via distance transform.
    For each slice containing myocardium, computes the distance-transform
    value (in mm, using in-plane spacing) at every myocardial pixel and
    takes 2x the mean distance-to-boundary as an approximate local
    thickness, then averages across slices. This is an approximation
    intended as a classification feature, not a validated clinical
    thickness measurement."""
    thicknesses = []
    for zi in range(mask3d.shape[2]):
        myo = (mask3d[:, :, zi] == class_id_myo)
        if myo.sum() == 0:
            continue
        dt = distance_transform_edt(myo, sampling=(sy, sx))
        thicknesses.append(2.0 * dt[myo].mean())
    if not thicknesses:
        return None
    return float(np.mean(thicknesses))


def body_surface_area_m2(height_cm, weight_kg):
    """DuBois & DuBois formula."""
    try:
        h = float(height_cm)
        w = float(weight_kg)
    except (TypeError, ValueError):
        return None
    if h <= 0 or w <= 0:
        return None
    return 0.007184 * (h ** 0.725) * (w ** 0.425)


def load_volume(path):
    img = nib.load(path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim == 4:  # some files keep a trailing singleton time dim
        data = data[..., 0]
    return data


def process_patient(model, row, img_size):
    sx = float(row["spacing_x_mm"]); sy = float(row["spacing_y_mm"]); sz = float(row["spacing_z_mm"])
    vox_ml = voxel_volume_ml(sx, sy, sz)

    ed_raw = load_volume(row["ed_image_path"])
    es_raw = load_volume(row["es_image_path"])
    ed_norm = normalize_volume(ed_raw)
    es_norm = normalize_volume(es_raw)

    ed_pred = segment_volume(model, ed_norm, img_size)
    es_pred = segment_volume(model, es_norm, img_size)

    lv_edv = class_volume_ml(ed_pred, CLASS_LV, vox_ml)
    lv_esv = class_volume_ml(es_pred, CLASS_LV, vox_ml)
    rv_edv = class_volume_ml(ed_pred, CLASS_RV, vox_ml)
    rv_esv = class_volume_ml(es_pred, CLASS_RV, vox_ml)
    myo_ed_vol = class_volume_ml(ed_pred, CLASS_MYO, vox_ml)
    myo_es_vol = class_volume_ml(es_pred, CLASS_MYO, vox_ml)

    lvef = 100.0 * (lv_edv - lv_esv) / lv_edv if lv_edv > 1e-6 else None
    rvef = 100.0 * (rv_edv - rv_esv) / rv_edv if rv_edv > 1e-6 else None
    myo_mass_ed = myo_ed_vol * MYOCARDIAL_DENSITY_G_PER_ML
    myo_mass_es = myo_es_vol * MYOCARDIAL_DENSITY_G_PER_ML
    wall_thickness_ed = mean_wall_thickness_mm(ed_pred, CLASS_MYO, sx, sy)

    bsa = body_surface_area_m2(row.get("height_cm"), row.get("weight_kg"))

    def indexed(v):
        return (v / bsa) if (bsa and v is not None) else None

    return {
        "patient_id": row["patient_id"],
        "patient": row["patient"],
        "split": row["split"],
        "group": row["group"],
        "lv_edv_ml": lv_edv, "lv_esv_ml": lv_esv, "lvef_pct": lvef,
        "rv_edv_ml": rv_edv, "rv_esv_ml": rv_esv, "rvef_pct": rvef,
        "myo_mass_ed_g": myo_mass_ed, "myo_mass_es_g": myo_mass_es,
        "wall_thickness_ed_mm": wall_thickness_ed,
        "bsa_m2": bsa,
        "lv_edv_indexed": indexed(lv_edv), "lv_esv_indexed": indexed(lv_esv),
        "rv_edv_indexed": indexed(rv_edv), "rv_esv_indexed": indexed(rv_esv),
        "myo_mass_ed_indexed": indexed(myo_mass_ed),
    }


def main(args):
    import csv as csv_mod
    with open(args.metadata_csv) as f:
        rows = list(csv_mod.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {args.metadata_csv}")

    model = load_model(args.network, args.run_id, args.model_root)

    out_rows = []
    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['patient']} ({row['group']}, split={row['split']})")
        try:
            out_rows.append(process_patient(model, row, args.img_size))
        except Exception as e:
            print(f"  FAILED: {e}")

    if not out_rows:
        raise RuntimeError("No biomarkers were extracted successfully.")

    fieldnames = list(out_rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n_missing_bsa = sum(1 for r in out_rows if r["bsa_m2"] is None)
    print(f"\nWrote {len(out_rows)} patients -> {args.out_csv}")
    if n_missing_bsa:
        print(f"Note: {n_missing_bsa} patients missing height/weight -> BSA-indexed features are blank for them.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv", required=True)
    p.add_argument("--run_id", required=True, help="Segmentation run_id, as produced by train_acdc.py")
    p.add_argument("--model_root", default="./model_pth_acdc")
    p.add_argument("--network", default="MK_UNet", choices=list(CONFIGS.keys()))
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--out_csv", default="./data/acdc_biomarkers.csv")
    main(p.parse_args())
