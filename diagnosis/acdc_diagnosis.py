from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

LABELS = ["NOR", "MINF", "DCM", "HCM", "RV"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
CLASS_IDS = {"RV": 1, "MYO": 2, "LV": 3}

META_COLUMNS = {
    "patient_id", "patient", "cohort", "label", "ed_frame", "es_frame", "feature_source"
}


def patient_id_from_name(name: str) -> int:
    return int(re.search(r"patient(\d{3})", name).group(1))


def discover_patient_dirs(root: Path) -> list[Path]:
    by_id = {}
    for path in Path(root).rglob("patient[0-9][0-9][0-9]"):
        if path.is_dir() and list(path.rglob("Info.cfg")):
            by_id[patient_id_from_name(path.name)] = path
    return [by_id[i] for i in sorted(by_id)]


def parse_info_cfg(path: Path) -> dict[str, str]:
    values = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def patient_metadata(patient_dir: Path) -> dict:
    cfg = parse_info_cfg(next(patient_dir.rglob("Info.cfg")))
    label = cfg["Group"].strip().upper().replace("ARV", "RV")
    height = float(cfg.get("Height", "nan"))
    weight = float(cfg.get("Weight", "nan"))
    return {
        "ed_frame": int(float(cfg["ED"])),
        "es_frame": int(float(cfg["ES"])),
        "label": label,
        "height_cm": height,
        "weight_kg": weight,
    }


def create_cohort_file(data_root: Path, output_csv: Path) -> pd.DataFrame:
    """Patients 1-100 are development; 101-150 are the untouched final test cohort."""
    rows = []
    for patient_dir in discover_patient_dirs(data_root):
        pid = patient_id_from_name(patient_dir.name)
        meta = patient_metadata(patient_dir)
        rows.append({
            "patient_id": pid,
            "patient": f"patient{pid:03d}",
            "cohort": "development" if pid <= 100 else "test",
            "label": meta["label"],
        })
    df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _cohort_map(cohort_csv: Path) -> dict[int, str]:
    df = pd.read_csv(cohort_csv)
    return dict(zip(df.patient_id.astype(int), df.cohort.astype(str)))


def _resolve_nii(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        return next(
            p for p in path.rglob("*")
            if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
        )
    raise FileNotFoundError(path)


def frame_path(patient_dir: Path, pid: int, frame: int, gt: bool = False) -> Path:
    suffix = "_gt" if gt else ""
    names = [
        f"patient{pid:03d}_frame{frame:02d}{suffix}.nii",
        f"patient{pid:03d}_frame{frame:02d}{suffix}.nii.gz",
    ]
    for name in names:
        direct = patient_dir / name
        if direct.exists():
            return _resolve_nii(direct)
        matches = list(patient_dir.rglob(name))
        if matches:
            return _resolve_nii(matches[0])
    raise FileNotFoundError(names[0])


def load_nifti(path: Path, dtype=np.float32):
    import nibabel as nib
    nii = nib.load(str(path))
    array = np.asarray(nii.get_fdata(), dtype=dtype)
    zooms = tuple(float(v) for v in nii.header.get_zooms()[:3])
    return array, zooms


def bsa_mosteller(height_cm: float, weight_kg: float) -> float:
    if not np.isfinite(height_cm) or not np.isfinite(weight_kg):
        return np.nan
    return math.sqrt(height_cm * weight_kg / 3600.0)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    finite = volume[np.isfinite(volume)]
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)
    volume = np.clip(volume, low, high)
    return ((volume - low) / (high - low)).astype(np.float32)


def _volume_ml(mask: np.ndarray, class_id: int, zooms) -> float:
    voxel_ml = float(np.prod(zooms)) / 1000.0
    return float(np.count_nonzero(mask == class_id) * voxel_ml)


def _ratio(a: float, b: float) -> float:
    return a / b if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-8 else np.nan


def _slice_contraction(ed: np.ndarray, es: np.ndarray, class_id: int) -> dict[str, float]:
    values = []
    for z in range(min(ed.shape[2], es.shape[2])):
        ed_area = float(np.count_nonzero(ed[:, :, z] == class_id))
        if ed_area >= 5:
            es_area = float(np.count_nonzero(es[:, :, z] == class_id))
            values.append((ed_area - es_area) / ed_area)
    if not values:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _segment_wall_thickness(
    mask: np.ndarray,
    zooms,
    n_sectors: int = 6,
    n_thirds: int = 3,
    ray_step: float = 0.5,
) -> dict[tuple[int, int], float]:
    """Myocardial wall thickness (mm) per (basal/mid/apical third, angular sector).

    For every short-axis slice that contains both LV cavity and myocardium, cast
    `n_sectors` rays outward from the LV centroid and measure how far each ray
    travels through MYO-labeled voxels. Slices are grouped into `n_thirds`
    contiguous groups (by stack position) to approximate base/mid/apex. This
    gives a coarse 2D map of regional wall thickness rather than a single
    global myocardial volume, which is what actually separates a regional
    disease (MINF: some segments thin/akinetic, others normal) from a global
    one (DCM: every segment is thin and dilated fairly uniformly).
    """
    lv_id, myo_id = CLASS_IDS["LV"], CLASS_IDS["MYO"]
    pix = float(np.mean(zooms[:2]))
    if not np.isfinite(pix) or pix <= 0:
        pix = 1.0

    valid_slices = [
        z for z in range(mask.shape[2])
        if np.count_nonzero(mask[:, :, z] == lv_id) >= 5
        and np.count_nonzero(mask[:, :, z] == myo_id) >= 5
    ]
    if not valid_slices:
        return {}

    thirds = np.array_split(valid_slices, min(n_thirds, len(valid_slices)))
    angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
    max_r = float(max(mask.shape[0], mask.shape[1]))
    steps = np.arange(0.0, max_r, ray_step)

    samples: dict[tuple[int, int], list[float]] = {}
    for third_idx, zs in enumerate(thirds):
        for z in zs:
            sl = mask[:, :, int(z)]
            ys, xs = np.nonzero(sl == lv_id)
            if len(xs) == 0:
                continue
            cx, cy = float(xs.mean()), float(ys.mean())
            h, w = sl.shape
            for sector_idx, angle in enumerate(angles):
                dx, dy = math.cos(angle), math.sin(angle)
                xi = np.clip(np.round(cx + dx * steps).astype(int), 0, w - 1)
                yi = np.clip(np.round(cy + dy * steps).astype(int), 0, h - 1)
                vals = sl[yi, xi]
                in_myo = vals == myo_id
                if not in_myo.any():
                    continue
                # thickness = contiguous run of MYO voxels the ray first crosses
                first_myo = int(np.argmax(in_myo))
                run_len = 0
                for v in in_myo[first_myo:]:
                    if not v:
                        break
                    run_len += 1
                if run_len == 0:
                    continue
                key = (third_idx, sector_idx)
                samples.setdefault(key, []).append(run_len * ray_step * pix)

    return {key: float(np.mean(vals)) for key, vals in samples.items()}


def _wall_thickness_features(ed_mask, es_mask, ed_zooms, es_zooms) -> dict[str, float]:
    """Summary regional wall-thickness/thickening biomarkers for MINF vs DCM.

    - *_wt_mean / *_wt_std / *_wt_min: mean, regional heterogeneity, and
      thinnest segment for ED/ES wall thickness. A low, near-uniform thickness
      with low std points toward global disease (DCM); a normal mean but high
      std / very low min points toward regional disease (MINF).
    - thickening_*: fractional systolic wall thickening per segment,
      (ES - ED) / ED. MINF segments over the infarct typically fail to
      thicken (near zero or negative) while remote segments thicken
      normally, producing high thickening_std and low thickening_min even
      when global EF looks only moderately reduced.
    """
    ed_seg = _segment_wall_thickness(ed_mask, ed_zooms)
    es_seg = _segment_wall_thickness(es_mask, es_zooms)
    shared_keys = sorted(set(ed_seg) & set(es_seg))

    def _stats(values: list[float], prefix: str) -> dict[str, float]:
        if not values:
            return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan, f"{prefix}_min": np.nan}
        arr = np.asarray(values, dtype=float)
        return {
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_std": float(arr.std()),
            f"{prefix}_min": float(arr.min()),
        }

    out = {}
    out.update(_stats(list(ed_seg.values()), "ed_wt"))
    out.update(_stats(list(es_seg.values()), "es_wt"))

    thickening = [
        (es_seg[k] - ed_seg[k]) / ed_seg[k]
        for k in shared_keys
        if ed_seg[k] > 1e-6
    ]
    out.update(_stats(thickening, "thickening"))
    out["n_wall_segments"] = float(len(shared_keys))
    return out


def cardiac_features(ed_mask, es_mask, ed_zooms, es_zooms, height_cm, weight_kg) -> dict:
    """Full segmentation-derived cardiac biomarker set used by Experiments 1 and 3.

    The same columns are computed from expert masks (Experiment 1) and MK-UNet
    predicted masks (Experiment 3), enabling a direct apples-to-apples comparison.
    """
    lv_edv = _volume_ml(ed_mask, CLASS_IDS["LV"], ed_zooms)
    lv_esv = _volume_ml(es_mask, CLASS_IDS["LV"], es_zooms)
    rv_edv = _volume_ml(ed_mask, CLASS_IDS["RV"], ed_zooms)
    rv_esv = _volume_ml(es_mask, CLASS_IDS["RV"], es_zooms)
    myo_edv = _volume_ml(ed_mask, CLASS_IDS["MYO"], ed_zooms)
    myo_esv = _volume_ml(es_mask, CLASS_IDS["MYO"], es_zooms)

    lv_sv = lv_edv - lv_esv
    rv_sv = rv_edv - rv_esv
    lv_ef = 100.0 * _ratio(lv_sv, lv_edv)
    rv_ef = 100.0 * _ratio(rv_sv, rv_edv)

    bsa = bsa_mosteller(height_cm, weight_kg)
    myo_mass = myo_edv * 1.05

    lv_con = _slice_contraction(ed_mask, es_mask, CLASS_IDS["LV"])
    rv_con = _slice_contraction(ed_mask, es_mask, CLASS_IDS["RV"])
    wall = _wall_thickness_features(ed_mask, es_mask, ed_zooms, es_zooms)

    return {
        "height_cm": float(height_cm),
        "weight_kg": float(weight_kg),
        "bsa_m2": float(bsa),
        "lv_edv_ml": lv_edv,
        "lv_esv_ml": lv_esv,
        "lv_sv_ml": lv_sv,
        "lv_ef_pct": lv_ef,
        "rv_edv_ml": rv_edv,
        "rv_esv_ml": rv_esv,
        "rv_sv_ml": rv_sv,
        "rv_ef_pct": rv_ef,
        "myo_edv_ml": myo_edv,
        "myo_esv_ml": myo_esv,
        "myo_mass_g": myo_mass,
        "lv_edvi_ml_m2": _ratio(lv_edv, bsa),
        "lv_esvi_ml_m2": _ratio(lv_esv, bsa),
        "rv_edvi_ml_m2": _ratio(rv_edv, bsa),
        "rv_esvi_ml_m2": _ratio(rv_esv, bsa),
        "myo_mass_g_m2": _ratio(myo_mass, bsa),
        "lv_rv_edv_ratio": _ratio(lv_edv, rv_edv),
        "myo_lv_ed_ratio": _ratio(myo_edv, lv_edv),
        "lv_slice_contraction_mean": lv_con["mean"],
        "lv_slice_contraction_std": lv_con["std"],
        "lv_slice_contraction_min": lv_con["min"],
        "lv_slice_contraction_max": lv_con["max"],
        "rv_slice_contraction_mean": rv_con["mean"],
        "rv_slice_contraction_std": rv_con["std"],
        **wall,
    }

def extract_gt_mask_features(data_root: Path, cohort_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Experiment 1: expert ACDC segmentation masks -> biomarkers."""
    cohort_map = _cohort_map(cohort_csv)
    rows = []
    for patient_dir in discover_patient_dirs(data_root):
        pid = patient_id_from_name(patient_dir.name)
        meta = patient_metadata(patient_dir)
        ed, ed_zooms = load_nifti(frame_path(patient_dir, pid, meta["ed_frame"], gt=True), np.uint8)
        es, es_zooms = load_nifti(frame_path(patient_dir, pid, meta["es_frame"], gt=True), np.uint8)
        rows.append({
            "patient_id": pid,
            "patient": f"patient{pid:03d}",
            "cohort": cohort_map[pid],
            "label": meta["label"],
            "feature_source": "ground_truth_masks",
            **cardiac_features(ed, es, ed_zooms, es_zooms, meta["height_cm"], meta["weight_kg"]),
        })
    df = pd.DataFrame(rows).sort_values("patient_id")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _grid_means(image: np.ndarray, grid: int = 4) -> dict[str, float]:
    out = {}
    for r, row_block in enumerate(np.array_split(image, grid, axis=0)):
        for c, block in enumerate(np.array_split(row_block, grid, axis=1)):
            out[f"g{r}{c}"] = float(block.mean())
    return out


def _raw_phase_features(volume: np.ndarray, prefix: str) -> dict[str, float]:
    volume = normalize_volume(volume)
    q10, q25, q50, q75, q90 = np.quantile(volume, [0.10, 0.25, 0.50, 0.75, 0.90])
    slice_means = volume.mean(axis=(0, 1))
    center = volume[:, :, volume.shape[2] // 2]
    features = {
        f"{prefix}_mean": float(volume.mean()),
        f"{prefix}_std": float(volume.std()),
        f"{prefix}_q10": float(q10),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q90": float(q90),
        f"{prefix}_slice_mean": float(slice_means.mean()),
        f"{prefix}_slice_std": float(slice_means.std()),
        f"{prefix}_slice_min": float(slice_means.min()),
        f"{prefix}_slice_max": float(slice_means.max()),
    }
    features.update({f"{prefix}_{k}": v for k, v in _grid_means(center).items()})
    return features


def extract_raw_mri_features(data_root: Path, cohort_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Experiment 2: raw ED/ES MRI summary features -> XGBoost. No segmentation masks."""
    cohort_map = _cohort_map(cohort_csv)
    rows = []
    for patient_dir in discover_patient_dirs(data_root):
        pid = patient_id_from_name(patient_dir.name)
        meta = patient_metadata(patient_dir)
        ed, _ = load_nifti(frame_path(patient_dir, pid, meta["ed_frame"], gt=False), np.float32)
        es, _ = load_nifti(frame_path(patient_dir, pid, meta["es_frame"], gt=False), np.float32)
        rows.append({
            "patient_id": pid,
            "patient": f"patient{pid:03d}",
            "cohort": cohort_map[pid],
            "label": meta["label"],
            "feature_source": "raw_mri_no_masks",
            "height_cm": meta["height_cm"],
            "weight_kg": meta["weight_kg"],
            "bsa_m2": bsa_mosteller(meta["height_cm"], meta["weight_kg"]),
            **_raw_phase_features(ed, "ed"),
            **_raw_phase_features(es, "es"),
        })
    df = pd.DataFrame(rows).sort_values("patient_id")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def build_mkunet(repo_dir: Path, checkpoint: Path, network: str = "MK_UNet"):
    import torch
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from mkunet_network import MK_UNet

    channels = {
        "MK_UNet_T": [4, 8, 16, 24, 32],
        "MK_UNet_S": [8, 16, 32, 48, 80],
        "MK_UNet": [16, 32, 64, 96, 160],
        "MK_UNet_M": [32, 64, 128, 192, 320],
        "MK_UNet_L": [64, 128, 256, 384, 512],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MK_UNet(
        num_classes=4,
        in_channels=1,
        channels=channels[network],
        deep_supervision=True,
    ).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device


def predict_volume(model, volume: np.ndarray, img_size: int, batch_size: int, device) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    volume = normalize_volume(volume)
    h, w, z = volume.shape
    slices = torch.from_numpy(np.moveaxis(volume, 2, 0)[:, None]).float()
    outputs = []
    with torch.no_grad():
        for start in range(0, z, batch_size):
            batch = slices[start:start + batch_size].to(device)
            batch = F.interpolate(batch, size=(img_size, img_size), mode="bilinear", align_corners=False)
            batch = (batch - 0.5) / 0.5
            logits = model(batch)[0]
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            outputs.append(torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy().astype(np.uint8))
    return np.moveaxis(np.concatenate(outputs, axis=0), 0, 2)


def _save_prediction_example(image: np.ndarray, mask: np.ndarray, output_png: Path, title: str):
    z = image.shape[2] // 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image[:, :, z], cmap="gray")
    axes[0].set_title("Cardiac MRI")
    axes[0].axis("off")
    axes[1].imshow(image[:, :, z], cmap="gray")
    axes[1].imshow(mask[:, :, z], alpha=0.45)
    axes[1].set_title("MK-UNet predicted mask")
    axes[1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def extract_mkunet_features(
    data_root: Path,
    cohort_csv: Path,
    output_csv: Path,
    repo_dir: Path,
    checkpoint: Path,
    example_png: Path | None = None,
    network: str = "MK_UNet",
    img_size: int = 224,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Experiment 3: MRI -> MK-UNet predicted masks -> biomarkers. No GT masks are opened."""
    cohort_map = _cohort_map(cohort_csv)
    model, device = build_mkunet(repo_dir, checkpoint, network)
    rows = []
    example_saved = False

    for patient_dir in discover_patient_dirs(data_root):
        pid = patient_id_from_name(patient_dir.name)
        meta = patient_metadata(patient_dir)
        ed_img, ed_zooms = load_nifti(frame_path(patient_dir, pid, meta["ed_frame"], gt=False), np.float32)
        es_img, es_zooms = load_nifti(frame_path(patient_dir, pid, meta["es_frame"], gt=False), np.float32)
        ed_pred = predict_volume(model, ed_img, img_size, batch_size, device)
        es_pred = predict_volume(model, es_img, img_size, batch_size, device)

        rows.append({
            "patient_id": pid,
            "patient": f"patient{pid:03d}",
            "cohort": cohort_map[pid],
            "label": meta["label"],
            "feature_source": "mkunet_predicted_masks",
            **cardiac_features(ed_pred, es_pred, ed_zooms, es_zooms, meta["height_cm"], meta["weight_kg"]),
        })

        if example_png is not None and not example_saved and cohort_map[pid] == "test":
            Path(example_png).parent.mkdir(parents=True, exist_ok=True)
            _save_prediction_example(
                normalize_volume(ed_img), ed_pred, Path(example_png), f"patient{pid:03d} — ED"
            )
            example_saved = True

    df = pd.DataFrame(rows).sort_values("patient_id")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _make_xgb(seed: int) -> XGBClassifier:
    # Applying strict regularization to prevent 100% training accuracy memorization.
    # NOTE: reg_alpha is intentionally lower than the previous config (1.0 -> 0.1)
    # at the user's request. Dropping n_estimators 100 -> 60 is the change most
    # likely to reduce train-set memorization on ~80-sample folds; if the CV train
    # accuracy is still pinned at 1.0 after this, raising reg_alpha/reg_lambda or
    # adding min_child_weight will do more than n_estimators/learning_rate alone.
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABELS),
        n_estimators=60,
        max_depth=3,            # shallow trees to avoid overfitting small sample sizes
        learning_rate=0.05,
        subsample=0.8,           # row sampling
        colsample_bytree=0.8,    # feature sampling
        reg_alpha=0.1,           # L1 regularization
        reg_lambda=1.0,          # L2 regularization
        random_state=seed,
        eval_metric="mlogloss",
        n_jobs=-1,
    )


def make_logreg_baseline(seed: int) -> Pipeline:
    """Linear baseline: standardize + multinomial logistic regression.

    Run through the exact same protocol (imputer, 5-fold CV, held-out test)
    as the XGBoost models. If XGBoost's improvement over this simple, low-
    variance linear model is small, that's a sign the biomarker features are
    carrying most of the signal and the gradient-boosted trees are adding
    little beyond what a linear model already captures -- useful context for
    judging whether XGBoost's apparent skill is real or partly small-sample
    overfitting.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            C=1.0,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a metric computed on paired (y_true, y_pred).

    Resamples patients with replacement `n_boot` times and recomputes the
    metric each time; returns the (alpha/2, 1 - alpha/2) percentiles. With
    only 50 independent-test patients, a single point accuracy is noisy --
    this quantifies that noise instead of implying more precision than the
    sample size supports.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    scores = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[b] = metric_fn(y_true[idx], y_pred[idx])
    low, high = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def _plot_confusion(y_true, y_pred, output_png: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_importance(model, columns: list[str], output_png: Path, top_n: int = 15):
    importance = pd.Series(model.feature_importances_, index=columns).sort_values().tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Final XGBoost feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_shap(model, X, feature_columns: list[str], output_dir: Path, prefix: str, max_display: int = 15):
    """SHAP interpretability for the final multiclass XGBoost model.

    Saves one overall mean(|SHAP|) importance chart (averaged across classes)
    plus one per-class beeswarm summary plot. `X` should already be imputed
    (e.g. the same array passed to model.predict/predict_proba).
    """
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_df = pd.DataFrame(np.asarray(X), columns=feature_columns)

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_df)

    # Older shap versions return a list of per-class (n_samples, n_features) arrays;
    # newer versions return one (n_samples, n_features, n_classes) array. Normalize.
    if isinstance(raw_shap, list):
        shap_per_class = raw_shap
    elif np.asarray(raw_shap).ndim == 3:
        shap_per_class = [raw_shap[:, :, c] for c in range(raw_shap.shape[2])]
    else:
        shap_per_class = [raw_shap]

    # Overall importance: mean |SHAP| averaged across all classes.
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_per_class], axis=0)
    overall = pd.Series(mean_abs, index=feature_columns).sort_values().tail(max_display)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    overall.plot(kind="barh", ax=ax)
    ax.set_title(f"SHAP feature importance — mean |SHAP| across classes ({prefix})")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(output_dir / f"shap_importance_overall_{prefix}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Per-class beeswarm summary (saved for the export bundle / closer inspection).
    for label, sv in zip(LABELS, shap_per_class):
        plt.figure()
        shap.summary_plot(sv, X_df, max_display=max_display, show=False)
        plt.title(f"SHAP summary — {label} ({prefix})")
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_{label}_{prefix}.png", dpi=180, bbox_inches="tight")
        plt.close()

    return overall


def _metric_dict(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _summary_from_fold_metrics(fold_metrics: pd.DataFrame, prefix: str) -> dict:
    return {
        metric: {
            "mean": float(fold_metrics[f"{prefix}_{metric}"].mean()),
            "std": float(fold_metrics[f"{prefix}_{metric}"].std(ddof=0)),
        }
        for metric in ["accuracy", "balanced_accuracy", "macro_f1"]
    }


def cross_validate_and_train(
    features_csv: Path,
    output_dir: Path,
    seed: int = 7,
    n_splits: int = 5,
    n_repeats: int = 1,
    model_factory=_make_xgb,
    model_name: str = "xgboost",
    n_bootstrap: int = 2000,
) -> dict:
    """Stratified CV on patients 1-100, then final held-out testing.

    For every fold we report BOTH training and validation metrics. After CV, a
    fresh final model is fitted on all development patients (1-100), its train
    performance is recorded, and it is evaluated once on patients 101-150.

    `n_repeats` controls how many times the 5-fold split is repeated with a
    different shuffle (via RepeatedStratifiedKFold) purely to make the CV
    mean/std estimate less noisy -- with ~80 training patients per fold, a
    single 5-fold run has high variance in which patients land in which fold.
    Repeating and averaging over more folds gives a more honest estimate of
    the model's spread without touching the untouched test cohort. The
    out-of-fold predictions used for `cv_predictions.csv` / the OOF confusion
    matrix still come from the first repeat only, so those artifacts are
    identical to the n_repeats=1 case.

    `model_factory(seed) -> unfitted estimator` lets the same protocol be
    reused for non-XGBoost models (e.g. a linear baseline) so pipelines are
    directly comparable. `n_bootstrap` sets the number of bootstrap resamples
    used for the 95% CI reported on the independent-test metrics; set to 0 to
    skip.
    """
    df = pd.read_csv(features_csv)
    feature_columns = [c for c in df.columns if c not in META_COLUMNS]
    development = df[df.cohort == "development"].copy().reset_index(drop=True)
    test = df[df.cohort == "test"].copy().reset_index(drop=True)

    y_dev = development.label.map(LABEL_TO_ID).to_numpy(dtype=int)
    if n_repeats > 1:
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_rows = []
    oof_rows = []
    oof_true = np.empty(len(development), dtype=int)
    oof_pred = np.empty(len(development), dtype=int)

    for split_i, (train_idx, val_idx) in enumerate(splitter.split(development, y_dev)):
        fold = split_i % n_splits + 1
        repeat = split_i // n_splits + 1
        fold_train = development.iloc[train_idx]
        fold_val = development.iloc[val_idx]

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(fold_train[feature_columns])
        X_val = imputer.transform(fold_val[feature_columns])
        y_train = fold_train.label.map(LABEL_TO_ID).to_numpy(dtype=int)
        y_val = fold_val.label.map(LABEL_TO_ID).to_numpy(dtype=int)

        model = model_factory(seed + split_i + 1)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train).astype(int)
        val_pred = model.predict(X_val).astype(int)
        train_scores = _metric_dict(y_train, train_pred)
        val_scores = _metric_dict(y_val, val_pred)

        fold_rows.append({
            "repeat": repeat,
            "fold": fold,
            **{f"train_{k}": v for k, v in train_scores.items()},
            **{f"val_{k}": v for k, v in val_scores.items()},
        })

        if repeat == 1:
            oof_true[val_idx] = y_val
            oof_pred[val_idx] = val_pred
            for row_i, pred_i in zip(val_idx, val_pred):
                row = development.iloc[row_i]
                oof_rows.append({
                    "fold": fold,
                    "patient_id": int(row.patient_id),
                    "patient": row.patient,
                    "label": row.label,
                    "predicted_label": ID_TO_LABEL[int(pred_i)],
                })

    fold_metrics = pd.DataFrame(fold_rows)
    cv_summary = {
        "train": _summary_from_fold_metrics(fold_metrics, "train"),
        "validation": _summary_from_fold_metrics(fold_metrics, "val"),
        "n_repeats": int(n_repeats),
        "n_folds_total": int(len(fold_metrics)),
    }

    # Final model: fit once on every development patient, then evaluate the untouched test cohort.
    final_imputer = SimpleImputer(strategy="median")
    X_dev = final_imputer.fit_transform(development[feature_columns])
    X_test = final_imputer.transform(test[feature_columns])
    y_test = test.label.map(LABEL_TO_ID).to_numpy(dtype=int)

    final_model = model_factory(seed)
    final_model.fit(X_dev, y_dev)

    final_train_pred = final_model.predict(X_dev).astype(int)
    final_train_metrics = _metric_dict(y_dev, final_train_pred)

    probability = final_model.predict_proba(X_test)
    test_pred = probability.argmax(axis=1)
    test_report = classification_report(
        y_test,
        test_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    test_metrics = {
        **_metric_dict(y_test, test_pred),
        "classification_report": test_report,
    }

    if n_bootstrap and n_bootstrap > 0 and len(y_test) > 0:
        macro_f1_fn = lambda yt, yp: f1_score(  # noqa: E731
            yt, yp, average="macro", labels=list(range(len(LABELS))), zero_division=0
        )
        test_metrics["accuracy_ci95"] = list(
            _bootstrap_ci(y_test, test_pred, accuracy_score, n_boot=n_bootstrap, seed=seed)
        )
        test_metrics["balanced_accuracy_ci95"] = list(
            _bootstrap_ci(y_test, test_pred, balanced_accuracy_score, n_boot=n_bootstrap, seed=seed + 1)
        )
        test_metrics["macro_f1_ci95"] = list(
            _bootstrap_ci(y_test, test_pred, macro_f1_fn, n_boot=n_bootstrap, seed=seed + 2)
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    pd.DataFrame(oof_rows).sort_values("patient_id").to_csv(
        output_dir / "cv_predictions.csv", index=False
    )

    test_predictions = test[["patient_id", "patient", "cohort", "label"]].copy()
    test_predictions["predicted_label"] = [ID_TO_LABEL[int(i)] for i in test_pred]
    test_predictions["correct"] = test_predictions.label == test_predictions.predicted_label
    for i, label in enumerate(LABELS):
        test_predictions[f"prob_{label}"] = probability[:, i]
    test_predictions.to_csv(output_dir / "predictions_test.csv", index=False)

    joblib.dump(
        {
            "model": final_model,
            "imputer": final_imputer,
            "feature_columns": feature_columns,
            "labels": LABELS,
        },
        output_dir / "model.joblib",
    )

    _plot_confusion(
        oof_true, oof_pred, output_dir / "confusion_cv_oof.png",
        "5-fold CV out-of-fold confusion"
    )
    _plot_confusion(
        y_test, test_pred, output_dir / "confusion_test.png",
        "Independent test confusion"
    )
    if hasattr(final_model, "feature_importances_"):
        _plot_importance(final_model, feature_columns, output_dir / "feature_importance.png")
        try:
            _plot_shap(final_model, X_test, feature_columns, output_dir, prefix="test")
        except ImportError:
            print("shap is not installed (pip install shap) — skipping SHAP analysis.")
        except Exception as exc:
            print(f"SHAP analysis skipped for {model_name}: {exc}")
    else:
        print(f"Skipping feature-importance/SHAP plots: '{model_name}' has no tree feature_importances_.")

    metrics = {
        "model_name": model_name,
        "cv": cv_summary,
        "final_train": final_train_metrics,
        "test": test_metrics,
        "n_development": int(len(development)),
        "n_test": int(len(test)),
        "n_splits": int(n_splits),
        "n_repeats": int(n_repeats),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def metrics_display_table(metrics: dict) -> pd.DataFrame:
    """Clear Train / Validation / Test summary for notebook display."""
    train_cv = metrics["cv"]["train"]
    val_cv = metrics["cv"]["validation"]

    return pd.DataFrame([
        {
            "Dataset": "5-Fold CV Train (mean ± std)",
            "Accuracy": f"{train_cv['accuracy']['mean']:.4f} ± {train_cv['accuracy']['std']:.4f}",
            "Balanced Accuracy": f"{train_cv['balanced_accuracy']['mean']:.4f} ± {train_cv['balanced_accuracy']['std']:.4f}",
            "Macro F1": f"{train_cv['macro_f1']['mean']:.4f} ± {train_cv['macro_f1']['std']:.4f}",
        },
        {
            "Dataset": "5-Fold CV Validation (mean ± std)",
            "Accuracy": f"{val_cv['accuracy']['mean']:.4f} ± {val_cv['accuracy']['std']:.4f}",
            "Balanced Accuracy": f"{val_cv['balanced_accuracy']['mean']:.4f} ± {val_cv['balanced_accuracy']['std']:.4f}",
            "Macro F1": f"{val_cv['macro_f1']['mean']:.4f} ± {val_cv['macro_f1']['std']:.4f}",
        },
        {
            "Dataset": "Final Train — patients 001–100",
            "Accuracy": f"{metrics['final_train']['accuracy']:.4f}",
            "Balanced Accuracy": f"{metrics['final_train']['balanced_accuracy']:.4f}",
            "Macro F1": f"{metrics['final_train']['macro_f1']:.4f}",
        },
        {
            "Dataset": "Independent Test — patients 101–150",
            "Accuracy": _with_ci(metrics["test"], "accuracy"),
            "Balanced Accuracy": _with_ci(metrics["test"], "balanced_accuracy"),
            "Macro F1": _with_ci(metrics["test"], "macro_f1"),
        },
    ])


def _with_ci(test_metrics: dict, key: str) -> str:
    """'{value:.4f}' or '{value:.4f} [lo, hi]' when a bootstrap CI was computed."""
    value = test_metrics[key]
    ci = test_metrics.get(f"{key}_ci95")
    if ci:
        return f"{value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
    return f"{value:.4f}"

def comparison_tables(experiments: dict[str, tuple[dict, Path]], output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    f1_rows = []
    for name, (metrics, _) in experiments.items():
        acc_ci = metrics["test"].get("accuracy_ci95", [np.nan, np.nan])
        f1_ci = metrics["test"].get("macro_f1_ci95", [np.nan, np.nan])
        rows.append({
            "Pipeline": name,
            "CV Train Accuracy Mean": metrics["cv"]["train"]["accuracy"]["mean"],
            "CV Train Accuracy Std": metrics["cv"]["train"]["accuracy"]["std"],
            "CV Val Accuracy Mean": metrics["cv"]["validation"]["accuracy"]["mean"],
            "CV Val Accuracy Std": metrics["cv"]["validation"]["accuracy"]["std"],
            "CV Train Macro F1 Mean": metrics["cv"]["train"]["macro_f1"]["mean"],
            "CV Train Macro F1 Std": metrics["cv"]["train"]["macro_f1"]["std"],
            "CV Val Macro F1 Mean": metrics["cv"]["validation"]["macro_f1"]["mean"],
            "CV Val Macro F1 Std": metrics["cv"]["validation"]["macro_f1"]["std"],
            "Final Train Accuracy": metrics["final_train"]["accuracy"],
            "Final Train Balanced Accuracy": metrics["final_train"]["balanced_accuracy"],
            "Final Train Macro F1": metrics["final_train"]["macro_f1"],
            "Test Accuracy": metrics["test"]["accuracy"],
            "Test Accuracy CI Low": acc_ci[0],
            "Test Accuracy CI High": acc_ci[1],
            "Test Balanced Accuracy": metrics["test"]["balanced_accuracy"],
            "Test Macro F1": metrics["test"]["macro_f1"],
            "Test Macro F1 CI Low": f1_ci[0],
            "Test Macro F1 CI High": f1_ci[1],
            "Number of Features": metrics["n_features"],
        })
        report = metrics["test"]["classification_report"]
        for label in LABELS:
            f1_rows.append({"Pipeline": name, "Class": label, "F1": report[label]["f1-score"]})

    comparison = pd.DataFrame(rows)
    per_class = pd.DataFrame(f1_rows)
    comparison.to_csv(output_dir / "pipeline_comparison.csv", index=False)
    per_class.to_csv(output_dir / "per_class_f1.csv", index=False)

    # Final held-out test comparison.
    ax = comparison.set_index("Pipeline")[[
        "Test Accuracy", "Test Balanced Accuracy", "Test Macro F1"
    ]].plot(kind="bar", figsize=(10, 5.5), rot=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Independent test-set comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "test_pipeline_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Test accuracy with bootstrap 95% CI — makes clear how much of any
    # between-pipeline gap is distinguishable given only ~50 test patients.
    if comparison["Test Accuracy CI Low"].notna().any():
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(comparison))
        acc = comparison["Test Accuracy"].to_numpy()
        lo = comparison["Test Accuracy CI Low"].to_numpy()
        hi = comparison["Test Accuracy CI High"].to_numpy()
        yerr = np.vstack([acc - lo, hi - acc])
        ax.errorbar(x, acc, yerr=yerr, fmt="o", capsize=5)
        ax.set_xticks(x, comparison["Pipeline"], rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Test accuracy")
        ax.set_title("Independent-test accuracy with 95% bootstrap CI")
        fig.tight_layout()
        fig.savefig(output_dir / "test_accuracy_ci.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # CV train-vs-validation comparison exposes overfitting directly.
    stage = comparison.set_index("Pipeline")[[
        "CV Train Macro F1 Mean", "CV Val Macro F1 Mean"
    ]].copy()
    stage.columns = ["CV Train Macro F1", "CV Validation Macro F1"]
    ax = stage.plot(kind="bar", figsize=(10, 5.5), rot=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("5-fold CV: training vs validation performance")
    plt.tight_layout()
    plt.savefig(output_dir / "cv_train_vs_validation.png", dpi=180, bbox_inches="tight")
    plt.close()

    # CV validation Macro-F1 with fold variability.
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comparison))
    ax.errorbar(
        x,
        comparison["CV Val Macro F1 Mean"],
        yerr=comparison["CV Val Macro F1 Std"],
        fmt="o",
        capsize=5,
    )
    ax.set_xticks(x, comparison["Pipeline"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("5-fold validation Macro-F1: mean ± standard deviation")
    fig.tight_layout()
    fig.savefig(output_dir / "cv_validation_macro_f1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pivot = per_class.pivot(index="Class", columns="Pipeline", values="F1").reindex(LABELS)
    ax = pivot.plot(kind="bar", figsize=(10, 5.5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Independent test per-class F1")
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_f1.png", dpi=180, bbox_inches="tight")
    plt.close()

    return comparison, pivot

def agreement_report(
    pred_csv_a: Path,
    pred_csv_b: Path,
    name_a: str,
    name_b: str,
    output_dir: Path,
) -> dict:
    """Agreement between two pipelines' independent-test predictions.

    Reuses the `predictions_test.csv` already written by `cross_validate_and_train`
    for each pipeline (e.g. GT-mask XGBoost vs MK-UNet XGBoost). Reports:
    - each pipeline's accuracy / Cohen's kappa against the true label
    - raw agreement rate and Cohen's kappa *between the two pipelines directly*
      (do the automatic and expert-derived biomarkers lead to the same
      diagnosis, independent of whether either is "correct"?)
    - a confusion matrix of pipeline A's predictions vs pipeline B's predictions

    A high inter-pipeline kappa is direct evidence that MK-UNet segmentations
    are an adequate substitute for expert masks in this diagnosis task -- the
    "automatic ~= expert" claim -- and is a stronger statement than simply
    citing two similar-looking test accuracies, since it's computed on
    matched patients rather than aggregate summary statistics.
    """
    a = pd.read_csv(pred_csv_a)[["patient", "label", "predicted_label"]].rename(
        columns={"predicted_label": "pred_a"}
    )
    b = pd.read_csv(pred_csv_b)[["patient", "predicted_label"]].rename(
        columns={"predicted_label": "pred_b"}
    )
    merged = a.merge(b, on="patient", how="inner").sort_values("patient")
    if len(merged) == 0:
        raise ValueError("No overlapping patients between the two prediction files.")

    y_true = merged["label"].to_numpy()
    pred_a = merged["pred_a"].to_numpy()
    pred_b = merged["pred_b"].to_numpy()

    acc_a = float(accuracy_score(y_true, pred_a))
    acc_b = float(accuracy_score(y_true, pred_b))
    kappa_a_vs_truth = float(cohen_kappa_score(y_true, pred_a))
    kappa_b_vs_truth = float(cohen_kappa_score(y_true, pred_b))

    agreement_rate = float((pred_a == pred_b).mean())
    kappa_a_vs_b = float(cohen_kappa_score(pred_a, pred_b))

    cm = confusion_matrix(pred_a, pred_b, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"{name_a}:{l}" for l in LABELS],
                          columns=[f"{name_b}:{l}" for l in LABELS])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "agreement_matched_predictions.csv", index=False)
    cm_df.to_csv(output_dir / "agreement_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel(f"{name_b} predicted label")
    ax.set_ylabel(f"{name_a} predicted label")
    ax.set_title(f"Pipeline agreement: {name_a} vs {name_b}\n"
                 f"(raw agreement {agreement_rate:.1%}, kappa {kappa_a_vs_b:.3f})")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "agreement_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    result = {
        "n_patients": int(len(merged)),
        "pipeline_a": name_a,
        "pipeline_b": name_b,
        "accuracy_a_vs_truth": acc_a,
        "accuracy_b_vs_truth": acc_b,
        "kappa_a_vs_truth": kappa_a_vs_truth,
        "kappa_b_vs_truth": kappa_b_vs_truth,
        "raw_agreement_a_vs_b": agreement_rate,
        "kappa_a_vs_b": kappa_a_vs_b,
    }
    (output_dir / "agreement_report.json").write_text(json.dumps(result, indent=2))
    return result


def export_bundle(
    results_dir: Path,
    diagnosis_dir: Path,
    checkpoint: Path,
    output_zip_base: Path,
    notebook_path: Path | None = None,
) -> Path:
    export_dir = Path(str(output_zip_base) + "_folder")
    if export_dir.exists():
        shutil.rmtree(export_dir)
    (export_dir / "models").mkdir(parents=True)
    (export_dir / "results").mkdir(parents=True)
    (export_dir / "code").mkdir(parents=True)

    shutil.copytree(results_dir, export_dir / "results", dirs_exist_ok=True)
    for filename in ["acdc_diagnosis.py", "README.md", "requirements.txt"]:
        source = diagnosis_dir / filename
        if source.exists():
            shutil.copy2(source, export_dir / "code" / filename)

    if notebook_path is not None and Path(notebook_path).exists():
        shutil.copy2(notebook_path, export_dir / Path(notebook_path).name)

    shutil.copy2(checkpoint, export_dir / "models" / checkpoint.name)
    for model in Path(results_dir).rglob("model.joblib"):
        shutil.copy2(model, export_dir / "models" / f"{model.parent.name}.joblib")

    return Path(shutil.make_archive(str(output_zip_base), "zip", export_dir))
