from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from .common import ID_TO_LABEL, LABELS, LABEL_TO_ID
except ImportError:
    from common import ID_TO_LABEL, LABELS, LABEL_TO_ID


def main(opt):
    bundle = joblib.load(opt.model)
    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_columns = bundle["feature_columns"]
    local_to_global = {int(k): int(v) for k, v in bundle.get("local_to_global", {}).items()}
    if not local_to_global:
        # Backward compatibility for older bundles trained with all five classes.
        local_to_global = {i: i for i in range(len(LABELS))}

    df = pd.read_csv(opt.features)
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    if opt.split.lower() != "all":
        df = df[df["split"] == opt.split].copy()
    if df.empty:
        raise RuntimeError(f"No rows available for split={opt.split!r}")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    y_true = df["label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    X = imputer.transform(df[feature_columns])
    prob_local = model.predict_proba(X)
    pred_local = np.asarray(prob_local).argmax(axis=1)
    y_pred = np.asarray([local_to_global[int(i)] for i in pred_local], dtype=int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true, y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }

    out_dir = Path(opt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pred_df = df[["patient_id", "patient", "split", "label"]].copy()
    pred_df["predicted_label"] = [ID_TO_LABEL[int(i)] for i in y_pred]
    pred_df["correct"] = pred_df["label"] == pred_df["predicted_label"]
    full_prob = np.zeros((len(df), len(LABELS)), dtype=float)
    for local_idx, global_idx in local_to_global.items():
        full_prob[:, int(global_idx)] = prob_local[:, int(local_idx)]
    for i, label in enumerate(LABELS):
        pred_df[f"prob_{label}"] = full_prob[:, i]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Diagnosis evaluation — {opt.split}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print("Saved:", out_dir.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a saved XGBoost diagnosis model on another ACDC feature CSV.")
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output_dir", required=True)
    main(p.parse_args())
