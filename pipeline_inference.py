"""
Full two-stage pipeline inference on a single new patient: raw ED/ES scans
in, predicted diagnosis out.

    Stage 1 (segmentation):  ED/ES volumes -> MK-UNet -> 3D masks
    Stage 2 (diagnosis):     masks -> clinical biomarkers -> Random Forest
                              -> {NOR, MINF, DCM, HCM, RV} + probabilities

Usage:
    python pipeline_inference.py \
        --patient_dir /path/to/patientXXX \
        --run_id ACDC_MK_UNet_bs16_sz224_lr0.0001_e150_seed42_120000 \
        --model_root ./model_pth_acdc \
        --diagnosis_model ./model_pth_acdc/diagnosis_classifier.joblib \
        --out_json ./predictions_acdc/patientXXX_diagnosis.json

Expects the same patientXXX/ layout as parse_acdc_metadata.py (an Info.cfg
with ED/ES frame indices, and the corresponding frame NIfTI files). Height
and weight are read from Info.cfg if present; pass --height_cm/--weight_kg
to override or supply them when Info.cfg lacks them (e.g. for a genuinely
new, unlabeled patient with no Info.cfg at all -- use --ed_frame/--es_frame
and --height_cm/--weight_kg directly in that case).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from extract_biomarkers import (
    body_surface_area_m2, class_volume_ml, load_model, load_volume,
    mean_wall_thickness_mm, normalize_volume, segment_volume, voxel_volume_ml,
    CLASS_LV, CLASS_RV, CLASS_MYO, MYOCARDIAL_DENSITY_G_PER_ML, CONFIGS,
)
from parse_acdc_metadata import parse_info_cfg, find_frame_image, get_spacing
from train_diagnosis_classifier import FEATURE_COLUMNS


def resolve_patient_inputs(patient_dir, ed_frame, es_frame, height_cm, weight_kg):
    patient_dir = Path(patient_dir)
    cfg_path = patient_dir / "Info.cfg"
    if cfg_path.exists():
        info = parse_info_cfg(cfg_path)
        ed_frame = ed_frame if ed_frame is not None else int(info["ED"])
        es_frame = es_frame if es_frame is not None else int(info["ES"])
        height_cm = height_cm if height_cm is not None else info.get("Height")
        weight_kg = weight_kg if weight_kg is not None else info.get("Weight")
    if ed_frame is None or es_frame is None:
        raise ValueError(
            "ED/ES frame indices not found. Provide --ed_frame/--es_frame "
            "explicitly, or make sure patient_dir contains an Info.cfg."
        )
    ed_path = find_frame_image(patient_dir, ed_frame)
    es_path = find_frame_image(patient_dir, es_frame)
    if ed_path is None or es_path is None:
        raise FileNotFoundError(f"Could not locate ED frame {ed_frame} / ES frame {es_frame} in {patient_dir}")
    return str(ed_path), str(es_path), height_cm, weight_kg


def compute_biomarkers(seg_model, ed_path, es_path, img_size, height_cm, weight_kg):
    sx, sy, sz = get_spacing(ed_path)
    vox_ml = voxel_volume_ml(sx, sy, sz)

    ed_norm = normalize_volume(load_volume(ed_path))
    es_norm = normalize_volume(load_volume(es_path))
    ed_pred = segment_volume(seg_model, ed_norm, img_size)
    es_pred = segment_volume(seg_model, es_norm, img_size)

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
    bsa = body_surface_area_m2(height_cm, weight_kg)

    def indexed(v):
        return (v / bsa) if (bsa and v is not None) else None

    return {
        "lv_edv_ml": lv_edv, "lv_esv_ml": lv_esv, "lvef_pct": lvef,
        "rv_edv_ml": rv_edv, "rv_esv_ml": rv_esv, "rvef_pct": rvef,
        "myo_mass_ed_g": myo_mass_ed, "myo_mass_es_g": myo_mass_es,
        "wall_thickness_ed_mm": wall_thickness_ed,
        "bsa_m2": bsa,
        "lv_edv_indexed": indexed(lv_edv), "lv_esv_indexed": indexed(lv_esv),
        "rv_edv_indexed": indexed(rv_edv), "rv_esv_indexed": indexed(rv_esv),
        "myo_mass_ed_indexed": indexed(myo_mass_ed),
    }


def predict_diagnosis(diagnosis_model_path, biomarkers):
    pipe = joblib.load(diagnosis_model_path)
    x = np.array([[biomarkers.get(c, np.nan) if biomarkers.get(c) is not None else np.nan
                    for c in FEATURE_COLUMNS]], dtype=np.float64)
    pred = pipe.predict(x)[0]
    proba = pipe.predict_proba(x)[0]
    classes = pipe.named_steps["clf"].classes_
    prob_by_class = {c: float(p) for c, p in zip(classes, proba)}
    return pred, prob_by_class


def main(args):
    ed_path, es_path, height_cm, weight_kg = resolve_patient_inputs(
        args.patient_dir, args.ed_frame, args.es_frame, args.height_cm, args.weight_kg
    )

    seg_model = load_model(args.network, args.run_id, args.model_root)
    biomarkers = compute_biomarkers(seg_model, ed_path, es_path, args.img_size, height_cm, weight_kg)
    diagnosis, probabilities = predict_diagnosis(args.diagnosis_model, biomarkers)

    result = {
        "patient_dir": str(args.patient_dir),
        "ed_image": ed_path,
        "es_image": es_path,
        "biomarkers": biomarkers,
        "predicted_diagnosis": diagnosis,
        "class_probabilities": probabilities,
    }

    print(json.dumps(result, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patient_dir", required=True)
    p.add_argument("--ed_frame", type=int, default=None)
    p.add_argument("--es_frame", type=int, default=None)
    p.add_argument("--height_cm", type=float, default=None)
    p.add_argument("--weight_kg", type=float, default=None)
    p.add_argument("--run_id", required=True, help="Segmentation run_id, as produced by train_acdc.py")
    p.add_argument("--model_root", default="./model_pth_acdc")
    p.add_argument("--network", default="MK_UNet", choices=list(CONFIGS.keys()))
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--diagnosis_model", default="./model_pth_acdc/diagnosis_classifier.joblib")
    p.add_argument("--out_json", default=None)
    main(p.parse_args())
