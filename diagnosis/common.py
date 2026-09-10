from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

CLASS_IDS = {"RV": 1, "MYO": 2, "LV": 3}
LABELS = ["NOR", "MINF", "DCM", "HCM", "RV"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

# Keep the classifier inputs centralized so GT and MK-UNet experiments use
# exactly the same variables.
FEATURE_COLUMNS = [
    "height_cm",
    "weight_kg",
    "bsa_m2",
    "lv_edv_ml",
    "lv_esv_ml",
    "lv_sv_ml",
    "lv_ef_pct",
    "rv_edv_ml",
    "rv_esv_ml",
    "rv_sv_ml",
    "rv_ef_pct",
    "myo_edv_ml",
    "myo_esv_ml",
    "myo_mass_g",
    "lv_edvi_ml_m2",
    "lv_esvi_ml_m2",
    "rv_edvi_ml_m2",
    "rv_esvi_ml_m2",
    "myo_mass_g_m2",
    "lv_rv_edv_ratio",
    "myo_lv_ed_ratio",
    "lv_slice_contraction_mean",
    "lv_slice_contraction_std",
    "lv_slice_contraction_min",
    "lv_slice_contraction_max",
    "rv_slice_contraction_mean",
    "rv_slice_contraction_std",
]


def split_for(pid: int, train_end: int = 80, val_end: int = 100) -> str:
    if pid <= train_end:
        return "train"
    if pid <= val_end:
        return "val"
    return "test"


def patient_id_from_name(name: str) -> int:
    m = re.search(r"patient(\d{3})", name)
    if not m:
        raise ValueError(f"Could not parse patient ID from {name}")
    return int(m.group(1))


def discover_patient_dirs(root: Path) -> List[Path]:
    """Return one useful directory per patient from the Kaggle ACDC mirror.

    The Kaggle mirror can contain odd nested entries whose names end in .nii,
    so selection is based on the patient directory rather than assuming one
    exact archive layout.
    """
    root = Path(root)
    candidates = [p for p in root.rglob("patient[0-9][0-9][0-9]") if p.is_dir()]
    by_pid: Dict[int, Path] = {}
    for p in candidates:
        pid = patient_id_from_name(p.name)
        # Prefer a candidate that contains Info.cfg; otherwise keep the first.
        score = int(any(x.is_file() for x in p.rglob("Info.cfg")))
        if pid not in by_pid:
            by_pid[pid] = p
        else:
            old = by_pid[pid]
            old_score = int(any(x.is_file() for x in old.rglob("Info.cfg")))
            if score > old_score:
                by_pid[pid] = p
    return [by_pid[k] for k in sorted(by_pid)]


def _resolve_file_or_nii_directory(entry: Path) -> Path:
    """Resolve Kaggle entries where a *.nii name can itself be a directory."""
    if entry.is_file():
        return entry
    if entry.is_dir():
        files = sorted([p for p in entry.rglob("*") if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))])
        if files:
            return files[0]
    raise FileNotFoundError(f"Could not resolve NIfTI entry: {entry}")


def find_info_cfg(patient_dir: Path) -> Path:
    direct = patient_dir / "Info.cfg"
    if direct.is_file():
        return direct
    matches = sorted(p for p in patient_dir.rglob("Info.cfg") if p.is_file())
    if not matches:
        raise FileNotFoundError(f"Info.cfg not found under {patient_dir}")
    return matches[0]


def parse_info_cfg(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in Path(path).read_text(errors="ignore").splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _candidate_frame_names(pid: int, frame: int, gt: bool) -> List[str]:
    suffix = "_gt" if gt else ""
    return [
        f"patient{pid:03d}_frame{frame:02d}{suffix}.nii",
        f"patient{pid:03d}_frame{frame:02d}{suffix}.nii.gz",
    ]


def find_frame_path(patient_dir: Path, pid: int, frame: int, gt: bool = False) -> Path:
    names = _candidate_frame_names(pid, frame, gt)
    for name in names:
        p = patient_dir / name
        if p.exists():
            return _resolve_file_or_nii_directory(p)
    for name in names:
        matches = sorted(patient_dir.rglob(name))
        for p in matches:
            try:
                return _resolve_file_or_nii_directory(p)
            except FileNotFoundError:
                pass
    raise FileNotFoundError(
        f"Could not find {'GT ' if gt else ''}frame {frame} for patient{pid:03d} under {patient_dir}"
    )


def metadata_for_patient(patient_dir: Path) -> Dict[str, object]:
    cfg = parse_info_cfg(find_info_cfg(patient_dir))
    required = ["ED", "ES", "Group"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing {missing} in {find_info_cfg(patient_dir)}")

    group = str(cfg["Group"]).strip().upper()
    # Some descriptions call this ARV; the ACDC Info.cfg label is commonly RV.
    if group == "ARV":
        group = "RV"
    if group not in LABEL_TO_ID:
        raise ValueError(f"Unexpected ACDC group '{group}' for {patient_dir.name}")

    def as_float(key: str) -> float:
        try:
            return float(cfg.get(key, "nan"))
        except Exception:
            return float("nan")

    return {
        "ed_frame": int(float(cfg["ED"])),
        "es_frame": int(float(cfg["ES"])),
        "label": group,
        "height_cm": as_float("Height"),
        "weight_kg": as_float("Weight"),
    }


def load_nifti(path: Path, dtype=np.float32) -> Tuple[np.ndarray, Tuple[float, float, float], object]:
    import nibabel as nib
    nii = nib.load(str(path))
    arr = np.asarray(nii.get_fdata(), dtype=dtype)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got shape {arr.shape}")
    zooms = tuple(float(x) for x in nii.header.get_zooms()[:3])
    return arr, zooms, nii


def bsa_mosteller(height_cm: float, weight_kg: float) -> float:
    if not np.isfinite(height_cm) or not np.isfinite(weight_kg) or height_cm <= 0 or weight_kg <= 0:
        return float("nan")
    return math.sqrt((height_cm * weight_kg) / 3600.0)


def class_volume_ml(mask: np.ndarray, class_id: int, zooms: Tuple[float, float, float]) -> float:
    voxel_ml = float(np.prod(np.asarray(zooms, dtype=np.float64))) / 1000.0
    return float(np.count_nonzero(mask == class_id) * voxel_ml)


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-8 else float("nan")


def ejection_fraction(edv: float, esv: float) -> float:
    return 100.0 * safe_ratio(edv - esv, edv)


def slice_contraction_stats(ed_mask: np.ndarray, es_mask: np.ndarray, class_id: int) -> Dict[str, float]:
    """Simple regional contraction descriptor from matched short-axis slices.

    Fractional area reduction is computed only where the class is present at ED.
    This is not a clinical wall-motion score; it is an interpretable research
    feature that can help distinguish globally vs regionally abnormal motion.
    """
    zmax = min(ed_mask.shape[2], es_mask.shape[2])
    values: List[float] = []
    for z in range(zmax):
        ed_area = float(np.count_nonzero(ed_mask[:, :, z] == class_id))
        if ed_area < 5:
            continue
        es_area = float(np.count_nonzero(es_mask[:, :, z] == class_id))
        values.append((ed_area - es_area) / ed_area)

    if not values:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def compute_cardiac_features(
    ed_mask: np.ndarray,
    es_mask: np.ndarray,
    ed_zooms: Tuple[float, float, float],
    es_zooms: Tuple[float, float, float],
    height_cm: float,
    weight_kg: float,
) -> Dict[str, float]:
    ed_mask = np.asarray(ed_mask, dtype=np.uint8)
    es_mask = np.asarray(es_mask, dtype=np.uint8)

    lv_edv = class_volume_ml(ed_mask, CLASS_IDS["LV"], ed_zooms)
    lv_esv = class_volume_ml(es_mask, CLASS_IDS["LV"], es_zooms)
    rv_edv = class_volume_ml(ed_mask, CLASS_IDS["RV"], ed_zooms)
    rv_esv = class_volume_ml(es_mask, CLASS_IDS["RV"], es_zooms)
    myo_edv = class_volume_ml(ed_mask, CLASS_IDS["MYO"], ed_zooms)
    myo_esv = class_volume_ml(es_mask, CLASS_IDS["MYO"], es_zooms)

    lv_sv = lv_edv - lv_esv
    rv_sv = rv_edv - rv_esv
    lv_ef = ejection_fraction(lv_edv, lv_esv)
    rv_ef = ejection_fraction(rv_edv, rv_esv)
    bsa = bsa_mosteller(height_cm, weight_kg)

    # Standard approximation used in cardiac MRI literature: myocardial density
    # around 1.05 g/mL. Kept explicit so it is easy to change in experiments.
    myo_mass = myo_edv * 1.05

    lv_con = slice_contraction_stats(ed_mask, es_mask, CLASS_IDS["LV"])
    rv_con = slice_contraction_stats(ed_mask, es_mask, CLASS_IDS["RV"])

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
        "lv_edvi_ml_m2": safe_ratio(lv_edv, bsa),
        "lv_esvi_ml_m2": safe_ratio(lv_esv, bsa),
        "rv_edvi_ml_m2": safe_ratio(rv_edv, bsa),
        "rv_esvi_ml_m2": safe_ratio(rv_esv, bsa),
        "myo_mass_g_m2": safe_ratio(myo_mass, bsa),
        "lv_rv_edv_ratio": safe_ratio(lv_edv, rv_edv),
        "myo_lv_ed_ratio": safe_ratio(myo_edv, lv_edv),
        "lv_slice_contraction_mean": lv_con["mean"],
        "lv_slice_contraction_std": lv_con["std"],
        "lv_slice_contraction_min": lv_con["min"],
        "lv_slice_contraction_max": lv_con["max"],
        "rv_slice_contraction_mean": rv_con["mean"],
        "rv_slice_contraction_std": rv_con["std"],
    }


def dice_for_class(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    p = pred == class_id
    g = gt == class_id
    den = p.sum() + g.sum()
    if den == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / den)
