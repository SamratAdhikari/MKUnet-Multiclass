from __future__ import annotations

import argparse
import json
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
from xgboost import XGBClassifier

try:
    from .common import FEATURE_COLUMNS, ID_TO_LABEL, LABELS, LABEL_TO_ID
except ImportError:
    from common import FEATURE_COLUMNS, ID_TO_LABEL, LABELS, LABEL_TO_ID


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_importance(model, feature_columns, path, top_n=15):
    imp = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax)
    ax.set_xlabel("XGBoost feature importance")
    ax.set_title(f"Top {min(top_n, len(imp))} diagnosis features")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(opt):
    df = pd.read_csv(opt.features)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    if "label" not in df.columns or "split" not in df.columns:
        raise KeyError("Feature CSV must contain 'label' and 'split' columns")

    df["label"] = df["label"].astype(str).str.upper().str.strip()
    unknown = sorted(set(df["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels in CSV: {unknown}")
    df["y"] = df["label"].map(LABEL_TO_ID).astype(int)

    out_dir = Path(opt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_splits = [x.strip() for x in opt.train_splits.split(",") if x.strip()]
    train_df = df[df["split"].isin(train_splits)].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError(
            f"Need non-empty train and test rows; counts={df['split'].value_counts().to_dict()}"
        )

    # This project is explicitly a five-class ACDC diagnosis experiment.  Fail
    # fast if a bad patient split removes a diagnosis category from training.
    missing_train_labels = sorted(set(LABELS) - set(train_df["label"]))
    if missing_train_labels:
        raise RuntimeError(
            f"Training split is missing ACDC diagnosis classes: {missing_train_labels}. "
            "Regenerate diagnosis_split.csv with create_diagnosis_split.py."
        )
    if not val_df.empty:
        missing_val_labels = sorted(set(LABELS) - set(val_df["label"]))
        if missing_val_labels:
            raise RuntimeError(
                f"Validation split is missing ACDC diagnosis classes: {missing_val_labels}. "
                "Regenerate diagnosis_split.csv with create_diagnosis_split.py."
            )
    missing_test_labels = sorted(set(LABELS) - set(test_df["label"]))
    if missing_test_labels:
        raise RuntimeError(
            f"Test split is missing ACDC diagnosis classes: {missing_test_labels}."
        )

    # XGBoost's sklearn API requires the labels used for fitting to be contiguous
    # 0..K-1.  A patient-ID split can occasionally omit one of the five ACDC
    # groups from the training subset, so map the *present* global ACDC class IDs
    # to local contiguous IDs for fitting. Predictions are mapped back afterwards.
    global_train_ids = sorted(int(x) for x in train_df["y"].unique())
    if len(global_train_ids) < 2:
        raise RuntimeError(
            "XGBoost diagnosis needs at least two classes in the training split. "
            f"Observed labels: {[ID_TO_LABEL[i] for i in global_train_ids]}"
        )
    global_to_local = {gid: i for i, gid in enumerate(global_train_ids)}
    local_to_global = {i: gid for gid, i in global_to_local.items()}

    print("Rows by split:", df["split"].value_counts().to_dict())
    print("Labels by split:\n", pd.crosstab(df["split"], df["label"]))
    print("Training classes:", [ID_TO_LABEL[i] for i in global_train_ids])

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_df[FEATURE_COLUMNS])
    y_train_global = train_df["y"].to_numpy(dtype=int)
    y_train_local = np.asarray([global_to_local[int(y)] for y in y_train_global], dtype=int)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(global_train_ids),
        n_estimators=opt.n_estimators,
        max_depth=opt.max_depth,
        learning_rate=opt.learning_rate,
        min_child_weight=opt.min_child_weight,
        subsample=opt.subsample,
        colsample_bytree=opt.colsample_bytree,
        reg_lambda=opt.reg_lambda,
        reg_alpha=opt.reg_alpha,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=opt.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_local)

    bundle = {
        "imputer": imputer,
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "labels": LABELS,
        "global_to_local": global_to_local,
        "local_to_global": local_to_global,
    }
    joblib.dump(bundle, out_dir / "xgboost_diagnosis.joblib")

    all_metrics = {
        "feature_source": str(df.get("feature_source", pd.Series(["unknown"])).iloc[0]),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_columns": FEATURE_COLUMNS,
        "training_classes": [ID_TO_LABEL[i] for i in global_train_ids],
        "xgboost_params": model.get_params(),
    }

    for split_name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if part.empty:
            continue
        X = imputer.transform(part[FEATURE_COLUMNS])
        y_true = part["y"].to_numpy(dtype=int)
        prob_local = model.predict_proba(X)
        pred_local = np.asarray(prob_local).argmax(axis=1)
        y_pred = np.asarray([local_to_global[int(i)] for i in pred_local], dtype=int)
        all_metrics[split_name] = evaluate(y_true, y_pred)

        pred_df = part[["patient_id", "patient", "split", "label"]].copy()
        pred_df["predicted_label"] = [ID_TO_LABEL[int(i)] for i in y_pred]
        pred_df["correct"] = pred_df["label"] == pred_df["predicted_label"]

        # Always emit probability columns for all five ACDC classes. Classes that
        # were absent during fit receive probability 0.0.
        full_prob = np.zeros((len(part), len(LABELS)), dtype=float)
        for local_idx, global_idx in local_to_global.items():
            full_prob[:, int(global_idx)] = prob_local[:, int(local_idx)]
        for i, label in enumerate(LABELS):
            pred_df[f"prob_{label}"] = full_prob[:, i]

        pred_df.to_csv(out_dir / f"predictions_{split_name}.csv", index=False)
        plot_confusion(
            y_true,
            y_pred,
            f"XGBoost diagnosis — {split_name}",
            out_dir / f"confusion_{split_name}.png",
        )

    plot_importance(model, FEATURE_COLUMNS, out_dir / "feature_importance.png", opt.top_features)
    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2, default=str))

    short_metrics = {
        k: {m: v for m, v in val.items() if m != "classification_report"}
        for k, val in all_metrics.items()
        if k in ("train", "val", "test") and isinstance(val, dict)
    }
    print(json.dumps(short_metrics, indent=2))
    print("Saved:", out_dir.resolve())
    print("metrics.json:", (out_dir / "metrics.json").resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train patient-level ACDC diagnosis with XGBoost.")
    p.add_argument("--features", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_splits", default="train", help="Comma-separated splits used to fit XGBoost; default=train")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n_estimators", type=int, default=250)
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=0.03)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--subsample", type=float, default=0.9)
    p.add_argument("--colsample_bytree", type=float, default=0.9)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--top_features", type=int, default=15)
    main(p.parse_args())
