from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .common import (
        compute_cardiac_features,
        discover_patient_dirs,
        find_frame_path,
        load_nifti,
        metadata_for_patient,
        patient_id_from_name,
        split_for,
    )
except ImportError:
    from common import (
        compute_cardiac_features,
        discover_patient_dirs,
        find_frame_path,
        load_nifti,
        metadata_for_patient,
        patient_id_from_name,
        split_for,
    )


def main(opt):
    data_root = Path(opt.data_root)
    patient_dirs = discover_patient_dirs(data_root)
    if not patient_dirs:
        raise RuntimeError(f"No patientXXX directories found under {data_root}")

    rows = []
    failures = []
    for patient_dir in tqdm(patient_dirs, desc="GT features"):
        pid = patient_id_from_name(patient_dir.name)
        if pid > opt.max_patient:
            continue
        try:
            meta = metadata_for_patient(patient_dir)
            ed_gt = find_frame_path(patient_dir, pid, meta["ed_frame"], gt=True)
            es_gt = find_frame_path(patient_dir, pid, meta["es_frame"], gt=True)
            ed_mask, ed_zooms, _ = load_nifti(ed_gt, dtype=np.uint8)
            es_mask, es_zooms, _ = load_nifti(es_gt, dtype=np.uint8)

            feats = compute_cardiac_features(
                ed_mask, es_mask, ed_zooms, es_zooms,
                float(meta["height_cm"]), float(meta["weight_kg"]),
            )
            rows.append({
                "patient_id": pid,
                "patient": f"patient{pid:03d}",
                "split": split_for(pid, opt.train_end, opt.val_end),
                "label": meta["label"],
                "ed_frame": meta["ed_frame"],
                "es_frame": meta["es_frame"],
                "feature_source": "ground_truth",
                **feats,
            })
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
    if failures:
        print(f"\nSkipped {len(failures)} patients:")
        for pid, msg in failures[:20]:
            print(f"  patient{pid:03d}: {msg}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract ACDC diagnosis features from ground-truth masks.")
    p.add_argument("--data_root", required=True, help="Root containing patientXXX folders from the Kaggle ACDC dataset")
    p.add_argument("--output", default="./diagnosis/results/gt_features.csv")
    p.add_argument("--train_end", type=int, default=80)
    p.add_argument("--val_end", type=int, default=100)
    p.add_argument("--max_patient", type=int, default=150)
    p.add_argument("--strict", action="store_true")
    main(p.parse_args())
