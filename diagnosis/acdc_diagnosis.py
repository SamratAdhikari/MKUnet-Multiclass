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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

LABELS = ["NOR", "MINF", "DCM", "HCM", "RV"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
CLASS_IDS = {"RV": 1, "MYO": 2, "LV": 3}

CARDIAC_FEATURES = [
    "height_cm", "weight_kg", "bsa_m2",
    "lv_edv_ml", "lv_esv_ml", "lv_sv_ml", "lv_ef_pct",
    "rv_edv_ml", "rv_esv_ml", "rv_sv_ml", "rv_ef_pct",
    "myo_edv_ml", "myo_esv_ml", "myo_mass_g",
    "lv_edvi_ml_m2", "lv_esvi_ml_m2",
    "rv_edvi_ml_m2", "rv_esvi_ml_m2",
    "myo_mass_g_m2", "lv_rv_edv_ratio", "myo_lv_ed_ratio",
    "lv_slice_contraction_mean", "lv_slice_contraction_std",
    "lv_slice_contraction_min", "lv_slice_contraction_max",
    "rv_slice_contraction_mean", "rv_slice_contraction_std",
]

META_COLUMNS = {
    "patient_id", "patient", "split", "label", "ed_frame", "es_frame",
    "feature_source",
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


def _resolve_nii(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        return next(p for p in path.rglob("*") if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")))
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
    return array, zooms, nii


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


def create_split(data_root: Path, output_csv: Path, seed: int = 7) -> pd.DataFrame:
    rows = []
    for patient_dir in discover_patient_dirs(data_root):
        pid = patient_id_from_name(patient_dir.name)
        meta = patient_metadata(patient_dir)
        rows.append({"patient_id": pid, "patient": f"patient{pid:03d}", "label": meta["label"]})

    df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    development = df[df.patient_id <= 100].copy()
    test = df[df.patient_id > 100].copy()

    train, val = train_test_split(
        development,
        test_size=0.20,
        random_state=seed,
        stratify=development["label"],
    )
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    split_df = pd.concat([train, val, test], ignore_index=True)
    split_df = split_df[["patient_id", "patient", "split", "label"]].sort_values("patient_id")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False)
    return split_df


def _split_map(split_csv: Path) -> dict[int, str]:
    df = pd.read_csv(split_csv)
    return dict(zip(df.patient_id.astype(int), df.split.astype(str)))


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


def cardiac_features(ed_mask, es_mask, ed_zooms, es_zooms, height_cm, weight_kg) -> dict:
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

    return {
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bsa_m2": bsa,
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
    }


def extract_gt_mask_features(data_root: Path, split_csv: Path, output_csv: Path) -> pd.DataFrame:
    split_map = _split_map(split_csv)
    rows = []
    for patient_dir in discover_patient_dirs(data_root):
        pid = patient_id_from_name(patient_dir.name)
        meta = patient_metadata(patient_dir)
        ed, ed_zooms, _ = load_nifti(frame_path(patient_dir, pid, meta["ed_frame"], gt=True), np.uint8)
        es, es_zooms, _ = load_nifti(frame_path(patient_dir, pid, meta["es_frame"], gt=True), np.uint8)
        rows.append({
            "patient_id": pid,
            "patient": f"patient{pid:03d}",
            "split": split_map[pid],
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


def extract_raw_mri_features(data_root: Path, split_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Mask-free XGBoost baseline: raw ED/ES MRI summary features + height/weight/BSA.

    No ground-truth segmentation mask and no MK-UNet prediction is used here.
    The ACDC Group field remains the supervised diagnosis target, not an input feature.
    """
    split_map = _split_map(split_csv)
    rows = []
    for patient_dir in discover_patient_dirs(data_root):
        pid = patient_id_from_name(patient_dir.name)
        meta = patient_metadata(patient_dir)
        ed, _, _ = load_nifti(frame_path(patient_dir, pid, meta["ed_frame"], gt=False), np.float32)
        es, _, _ = load_nifti(frame_path(patient_dir, pid, meta["es_frame"], gt=False), np.float32)
        ed_features = _raw_phase_features(ed, "ed")
        es_features = _raw_phase_features(es, "es")
        bsa = bsa_mosteller(meta["height_cm"], meta["weight_kg"])
        rows.append({
            "patient_id": pid,
            "patient": f"patient{pid:03d}",
            "split": split_map[pid],
            "label": meta["label"],
            "feature_source": "raw_mri_no_masks",
            "height_cm": meta["height_cm"],
            "weight_kg": meta["weight_kg"],
            "bsa_m2": bsa,
            **ed_features,
            **es_features,
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
    split_csv: Path,
    output_csv: Path,
    repo_dir: Path,
    checkpoint: Path,
    example_png: Path | None = None,
    network: str = "MK_UNet",
    img_size: int = 224,
    batch_size: int = 8,
) -> pd.DataFrame:
    """MRI -> MK-UNet predicted masks -> cardiac biomarkers.

    Ground-truth segmentation masks are never opened by this function.
    """
    split_map = _split_map(split_csv)
    model, device = build_mkunet(repo_dir, checkpoint, network)
    rows = []
    example_saved = False

    for patient_dir in discover_patient_dirs(data_root):
        pid = patient_id_from_name(patient_dir.name)
        meta = patient_metadata(patient_dir)
        ed_img, ed_zooms, _ = load_nifti(frame_path(patient_dir, pid, meta["ed_frame"], gt=False), np.float32)
        es_img, es_zooms, _ = load_nifti(frame_path(patient_dir, pid, meta["es_frame"], gt=False), np.float32)
        ed_pred = predict_volume(model, ed_img, img_size, batch_size, device)
        es_pred = predict_volume(model, es_img, img_size, batch_size, device)

        rows.append({
            "patient_id": pid,
            "patient": f"patient{pid:03d}",
            "split": split_map[pid],
            "label": meta["label"],
            "feature_source": "mkunet_predicted_masks",
            **cardiac_features(ed_pred, es_pred, ed_zooms, es_zooms, meta["height_cm"], meta["weight_kg"]),
        })

        if example_png is not None and not example_saved and split_map[pid] == "test":
            Path(example_png).parent.mkdir(parents=True, exist_ok=True)
            _save_prediction_example(normalize_volume(ed_img), ed_pred, Path(example_png), f"patient{pid:03d} — ED")
            example_saved = True

    df = pd.DataFrame(rows).sort_values("patient_id")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


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
    ax.set_title("XGBoost feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_xgboost(features_csv: Path, output_dir: Path, seed: int = 7) -> dict:
    df = pd.read_csv(features_csv)
    feature_columns = [c for c in df.columns if c not in META_COLUMNS]
    train = df[df.split == "train"].copy()

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train[feature_columns])
    y_train = train.label.map(LABEL_TO_ID).to_numpy(dtype=int)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABELS),
        n_estimators=250,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "imputer": imputer, "feature_columns": feature_columns},
        output_dir / "model.joblib",
    )

    all_metrics = {}
    for split_name in ["train", "val", "test"]:
        part = df[df.split == split_name].copy()
        X = imputer.transform(part[feature_columns])
        y_true = part.label.map(LABEL_TO_ID).to_numpy(dtype=int)
        probability = model.predict_proba(X)
        y_pred = probability.argmax(axis=1)

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        )
        all_metrics[split_name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "classification_report": report,
        }

        prediction = part[["patient_id", "patient", "split", "label"]].copy()
        prediction["predicted_label"] = [ID_TO_LABEL[i] for i in y_pred]
        prediction["correct"] = prediction.label == prediction.predicted_label
        for i, label in enumerate(LABELS):
            prediction[f"prob_{label}"] = probability[:, i]
        prediction.to_csv(output_dir / f"predictions_{split_name}.csv", index=False)

        _plot_confusion(
            y_true,
            y_pred,
            output_dir / f"confusion_{split_name}.png",
            f"{split_name.title()} confusion matrix",
        )

    _plot_importance(model, feature_columns, output_dir / "feature_importance.png")
    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    return all_metrics


def comparison_tables(experiments: dict[str, tuple[dict, Path]], output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    f1_rows = []
    for name, (metrics, _) in experiments.items():
        test = metrics["test"]
        rows.append({
            "Pipeline": name,
            "Accuracy": test["accuracy"],
            "Balanced Accuracy": test["balanced_accuracy"],
            "Macro F1": test["macro_f1"],
        })
        report = test["classification_report"]
        for label in LABELS:
            f1_rows.append({"Pipeline": name, "Class": label, "F1": report[label]["f1-score"]})

    comparison = pd.DataFrame(rows)
    per_class = pd.DataFrame(f1_rows)
    comparison.to_csv(output_dir / "pipeline_comparison.csv", index=False)
    per_class.to_csv(output_dir / "per_class_f1.csv", index=False)

    ax = comparison.set_index("Pipeline")[["Accuracy", "Balanced Accuracy", "Macro F1"]].plot(
        kind="bar", figsize=(10, 5.5), rot=15
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("ACDC diagnosis pipeline comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "pipeline_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()

    pivot = per_class.pivot(index="Class", columns="Pipeline", values="F1").reindex(LABELS)
    ax = pivot.plot(kind="bar", figsize=(10, 5.5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_f1.png", dpi=180, bbox_inches="tight")
    plt.close()

    return comparison, pivot


def export_bundle(
    results_dir: Path,
    diagnosis_dir: Path,
    checkpoint: Path,
    output_zip_base: Path,
) -> Path:
    export_dir = Path(str(output_zip_base) + "_folder")
    if export_dir.exists():
        shutil.rmtree(export_dir)
    (export_dir / "models").mkdir(parents=True)
    (export_dir / "results").mkdir(parents=True)
    (export_dir / "code").mkdir(parents=True)

    shutil.copytree(results_dir, export_dir / "results", dirs_exist_ok=True)
    shutil.copy2(diagnosis_dir / "acdc_diagnosis.py", export_dir / "code" / "acdc_diagnosis.py")
    shutil.copy2(checkpoint, export_dir / "models" / checkpoint.name)

    for model in Path(results_dir).rglob("model.joblib"):
        shutil.copy2(model, export_dir / "models" / f"{model.parent.name}.joblib")

    zip_file = shutil.make_archive(str(output_zip_base), "zip", export_dir)
    return Path(zip_file)
