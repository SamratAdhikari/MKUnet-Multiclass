from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .common import LABELS, discover_patient_dirs, metadata_for_patient, patient_id_from_name
except ImportError:
    from common import LABELS, discover_patient_dirs, metadata_for_patient, patient_id_from_name


def validate_split_table(df: pd.DataFrame) -> None:
    required = {"patient_id", "patient", "split", "label"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise KeyError(f"Split table is missing columns: {sorted(missing_cols)}")

    if df["patient_id"].duplicated().any():
        dupes = df.loc[df["patient_id"].duplicated(), "patient_id"].tolist()
        raise ValueError(f"Duplicate patient IDs in split table: {dupes}")

    expected_splits = {"train", "val", "test"}
    unknown_splits = set(df["split"]) - expected_splits
    if unknown_splits:
        raise ValueError(f"Unknown split names: {sorted(unknown_splits)}")

    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        missing_labels = sorted(set(LABELS) - set(part["label"]))
        if missing_labels:
            raise RuntimeError(
                f"Split '{split}' is missing diagnosis classes {missing_labels}. "
                "The diagnosis experiment requires all five ACDC groups in each split."
            )


def main(opt):
    data_root = Path(opt.data_root)
    patient_dirs = discover_patient_dirs(data_root)
    if not patient_dirs:
        raise RuntimeError(f"No patientXXX directories found under {data_root}")

    rows = []
    for patient_dir in patient_dirs:
        pid = patient_id_from_name(patient_dir.name)
        if pid > opt.max_patient:
            continue
        meta = metadata_for_patient(patient_dir)
        rows.append({
            "patient_id": pid,
            "patient": f"patient{pid:03d}",
            "label": meta["label"],
        })

    df = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No ACDC patients were discovered.")

    development = df[df["patient_id"] <= opt.development_end].copy()
    final_test = df[df["patient_id"] > opt.development_end].copy()

    if development.empty or final_test.empty:
        raise RuntimeError(
            f"Expected both development (<= {opt.development_end}) and final-test "
            f"(> {opt.development_end}) patients; counts: dev={len(development)}, test={len(final_test)}"
        )

    train_df, val_df = train_test_split(
        development,
        test_size=opt.val_fraction,
        random_state=opt.seed,
        shuffle=True,
        stratify=development["label"],
    )

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    final_test = final_test.copy(); final_test["split"] = "test"

    split_df = pd.concat([train_df, val_df, final_test], ignore_index=True)
    split_df = split_df[["patient_id", "patient", "split", "label"]].sort_values("patient_id").reset_index(drop=True)

    validate_split_table(split_df)

    out = Path(opt.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(out, index=False)

    print(f"Saved diagnosis split to: {out}")
    print(f"Seed: {opt.seed}")
    print(f"Development cohort: patient001-patient{opt.development_end:03d}")
    print(f"Final test cohort: patient{opt.development_end + 1:03d}-patient{opt.max_patient:03d}")
    print("\nClass distribution:")
    print(pd.crosstab(split_df["split"], split_df["label"]).reindex(columns=LABELS, fill_value=0))
    print("\nPatient counts:", split_df["split"].value_counts().to_dict())


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "Create one deterministic stratified patient split for ACDC diagnosis. "
            "Patients 1-development_end are randomly stratified into train/val; "
            "patients above development_end remain the untouched final test cohort."
        )
    )
    p.add_argument("--data_root", required=True)
    p.add_argument("--output", default="./diagnosis/results/diagnosis_split.csv")
    p.add_argument("--development_end", type=int, default=100)
    p.add_argument("--max_patient", type=int, default=150)
    p.add_argument("--val_fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=7)
    main(p.parse_args())
