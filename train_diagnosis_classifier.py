"""
Stage 2 of the two-stage diagnosis pipeline: train a classifier that maps
segmentation-derived clinical biomarkers -> diagnosis class
(NOR / MINF / DCM / HCM / RV).

Given how few patients ACDC has (100 labeled), a single train/test split is
noisy, so this script reports stratified k-fold cross-validation accuracy
and macro-F1 as the primary metric, then refits on all available data and
saves that final model for use in pipeline_inference.py.

If a 'test' split exists in the biomarkers CSV (see the WARNING printed by
parse_acdc_metadata.py about this), it is also evaluated separately and
held out of cross-validation, to mirror how the segmentation model itself
was evaluated.

Usage:
    python train_diagnosis_classifier.py \
        --biomarkers_csv ./data/acdc_biomarkers.csv \
        --out_model ./model_pth_acdc/diagnosis_classifier.joblib \
        --n_splits 5
"""
import argparse
import csv
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "lv_edv_ml", "lv_esv_ml", "lvef_pct",
    "rv_edv_ml", "rv_esv_ml", "rvef_pct",
    "myo_mass_ed_g", "myo_mass_es_g", "wall_thickness_ed_mm",
    "lv_edv_indexed", "lv_esv_indexed", "rv_edv_indexed", "rv_esv_indexed",
    "myo_mass_ed_indexed",
]


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def to_xy(rows):
    X, y = [], []
    for r in rows:
        feats = []
        for col in FEATURE_COLUMNS:
            v = r.get(col, "")
            feats.append(float(v) if v not in ("", None) else np.nan)
        X.append(feats)
        y.append(r["group"])
    return np.array(X, dtype=np.float64), np.array(y)


def build_pipeline(seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=seed,
        )),
    ])


def cross_validate(X, y, n_splits, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s = [], []
    all_true, all_pred = [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        pipe = build_pipeline(seed)
        pipe.fit(X[tr_idx], y[tr_idx])
        pred = pipe.predict(X[te_idx])
        acc = accuracy_score(y[te_idx], pred)
        f1 = f1_score(y[te_idx], pred, average="macro")
        accs.append(acc); f1s.append(f1)
        all_true.extend(y[te_idx]); all_pred.extend(pred)
        print(f"  fold {fold}: acc={acc:.3f} macro-F1={f1:.3f} (n_test={len(te_idx)})")
    return accs, f1s, all_true, all_pred


def main(args):
    rows = load_rows(args.biomarkers_csv)
    if not rows:
        raise RuntimeError(f"No rows in {args.biomarkers_csv}")

    train_val_rows = [r for r in rows if r["split"] in ("train", "val")]
    test_rows = [r for r in rows if r["split"] == "test"]
    if not train_val_rows:
        print("No 'split' column values of train/val found; using all rows for CV.")
        train_val_rows = rows

    X, y = to_xy(train_val_rows)
    classes = sorted(set(y))
    print(f"Training pool: {len(y)} patients, classes={classes}")
    counts = {c: int((y == c).sum()) for c in classes}
    print("Class counts:", counts)

    n_splits = min(args.n_splits, min(counts.values()))
    if n_splits < args.n_splits:
        print(f"Reducing n_splits to {n_splits} (smallest class has only {n_splits} samples).")

    print(f"\n{n_splits}-fold stratified cross-validation:")
    accs, f1s, all_true, all_pred = cross_validate(X, y, n_splits, args.seed)
    print(f"\nMean CV accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"Mean CV macro-F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    print("\nAggregated classification report (across folds):")
    print(classification_report(all_true, all_pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), label order:", classes)
    print(confusion_matrix(all_true, all_pred, labels=classes))

    # Refit on all train/val data for the deployed model.
    final_pipe = build_pipeline(args.seed)
    final_pipe.fit(X, y)

    result = {
        "cv_mean_accuracy": float(np.mean(accs)),
        "cv_mean_macro_f1": float(np.mean(f1s)),
        "n_splits": n_splits,
        "classes": classes,
        "feature_columns": FEATURE_COLUMNS,
    }

    if test_rows:
        Xt, yt = to_xy(test_rows)
        test_pred = final_pipe.predict(Xt)
        test_acc = accuracy_score(yt, test_pred)
        test_f1 = f1_score(yt, test_pred, average="macro")
        print(f"\nHeld-out test split ({len(yt)} patients): acc={test_acc:.3f} macro-F1={test_f1:.3f}")
        print(classification_report(yt, test_pred, zero_division=0))
        result["test_accuracy"] = float(test_acc)
        result["test_macro_f1"] = float(test_f1)
    else:
        print("\nNo 'test' split patients found -- skipping held-out test evaluation "
              "(see the WARNING from parse_acdc_metadata.py).")

    importances = final_pipe.named_steps["clf"].feature_importances_
    ranked = sorted(zip(FEATURE_COLUMNS, importances), key=lambda t: -t[1])
    print("\nFeature importances (Random Forest):")
    for name, imp in ranked:
        print(f"  {name:24s} {imp:.3f}")
    result["feature_importances"] = {name: float(imp) for name, imp in ranked}

    joblib.dump(final_pipe, args.out_model)
    print(f"\nSaved final classifier -> {args.out_model}")

    with open(args.out_metrics, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved metrics -> {args.out_metrics}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--biomarkers_csv", required=True)
    p.add_argument("--out_model", default="./model_pth_acdc/diagnosis_classifier.joblib")
    p.add_argument("--out_metrics", default="./model_pth_acdc/diagnosis_classifier_metrics.json")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
