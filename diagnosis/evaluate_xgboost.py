from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score

try:
    from .common import ID_TO_LABEL, LABELS, LABEL_TO_ID
except ImportError:
    from common import ID_TO_LABEL, LABELS, LABEL_TO_ID


def main(opt):
    bundle = joblib.load(opt.model)
    model = bundle["model"]
    imputer = bundle["imputer"]
    features = bundle["feature_columns"]

    df = pd.read_csv(opt.features)
    df["label"] = df["label"].astype(str).str.upper().replace({"ARV": "RV"})
    part = df[df["split"] == opt.split].copy()
    if part.empty:
        raise RuntimeError(f"No rows for split={opt.split}")

    y_true = part["label"].map(LABEL_TO_ID).to_numpy()
    X = imputer.transform(part[features])
    prob = model.predict_proba(X)
    y_pred = prob.argmax(axis=1)

    metrics = {
        "split": opt.split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true, y_pred,
            labels=list(range(len(LABELS))), target_names=LABELS,
            output_dict=True, zero_division=0,
        ),
    }

    out_dir = Path(opt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = part[["patient_id", "patient", "split", "label"]].copy()
    pred_df["predicted_label"] = [ID_TO_LABEL[int(i)] for i in y_pred]
    pred_df["correct"] = pred_df["label"] == pred_df["predicted_label"]
    for i, label in enumerate(LABELS):
        pred_df[f"prob_{label}"] = prob[:, i]
    pred_df.to_csv(out_dir / f"predictions_{opt.split}.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a saved XGBoost diagnosis model on another feature source.")
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--output_dir", required=True)
    main(p.parse_args())
