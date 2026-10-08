"""Compare natural, weighted and 50/50 undersampled logistic models.

Chronological 64/16/20 train/validation/test split. Validation selects the
decision threshold; test labels are never used for model or threshold selection.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classification_data import TARGET_COLUMN, DELIVERED_MODEL_FEATURES, build_delivered_order_classification_dataset, split_delivered_order_features_target
from src.classification_model import build_delivered_order_logistic_pipeline

orders = pd.read_csv(ROOT / "data/processed/order_level.csv")
items = pd.read_csv(ROOT / "data/processed/item_level.csv")
df = build_delivered_order_classification_dataset(orders, items)
lookup = orders.set_index("order_id").loc[df["order_id"]]
df["purchase_time"] = pd.to_datetime(lookup["order_purchase_timestamp"], format="mixed").to_numpy()
reviewed_at = pd.to_datetime(lookup["review_answer_timestamp"], format="mixed").to_numpy()
delivered_at = pd.to_datetime(lookup["order_delivered_customer_date"], format="mixed").to_numpy()
df = df.loc[~(reviewed_at < delivered_at)].sort_values("purchase_time", kind="stable").reset_index(drop=True)
X, y = split_delivered_order_features_target(df)
assert X.columns.tolist() == DELIVERED_MODEL_FEATURES
n = len(y)
train_end, test_start = int(n * .64), int(n * .80)
X_fit, y_fit = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:test_start], y.iloc[train_end:test_start]
X_full, y_full = X.iloc[:test_start], y.iloc[:test_start]
X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]
assert all(s.nunique() == 2 for s in (y_fit, y_val, y_test)), "Both classes required in each split"
rows = []
for name in ("natural", "class_weight_balanced", "undersample_50_50"):
    def fit_data(X_source, y_source):
        if name != "undersample_50_50":
            return X_source, y_source
        rng = np.random.default_rng(42)
        counts = y_source.value_counts()
        size = int(counts.min())
        selected = np.sort(np.concatenate([rng.choice(np.flatnonzero(y_source.to_numpy() == cls), size=size, replace=False) for cls in (0, 1)]))
        return X_source.iloc[selected], y_source.iloc[selected]

    def pipeline():
        p = build_delivered_order_logistic_pipeline(random_state=42)
        p.set_params(classifier__class_weight="balanced" if name == "class_weight_balanced" else None)
        return p

    x, yy = fit_data(X_fit, y_fit)
    model = pipeline().fit(x, yy)
    val_probs = model.predict_proba(X_val)[:, 1]
    # Choose threshold on validation only, maximizing F1. Tie-break to higher threshold.
    candidates = np.arange(.01, 1., .01)
    threshold = float(max(candidates, key=lambda t: (f1_score(y_val, val_probs >= t, zero_division=0), t)))
    x, yy = fit_data(X_full, y_full)
    model = pipeline().fit(x, yy)
    probs = model.predict_proba(X_test)[:, 1]
    for cutoff_label, cutoff in (("default_0.50", .5), ("validation_f1", threshold)):
        pred = probs >= cutoff
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rows.append(dict(approach=name, policy=cutoff_label, threshold=round(cutoff, 3),
            train_rows=len(yy), test_rows=len(y_test), test_negative_rate=round(float(y_test.mean()), 4),
            precision=round(precision_score(y_test, pred, zero_division=0), 4),
            recall=round(recall_score(y_test, pred, zero_division=0), 4),
            f1=round(f1_score(y_test, pred, zero_division=0), 4),
            pr_auc=round(average_precision_score(y_test, probs), 4),
            roc_auc=round(roc_auc_score(y_test, probs), 4),
            flagged=int(pred.sum()), false_alarms=int(fp), missed_negatives=int(fn), detected_negatives=int(tp)))

out = ROOT / "reports/phase2"
out.mkdir(parents=True, exist_ok=True)
result = pd.DataFrame(rows)
result.to_csv(out / "three_approaches.csv", index=False)
summary = (f"## Three approaches, chronological holdout\n\nEligible orders: {n:,}. "
           f"Train: {train_end:,}, validation: {test_start-train_end:,}, test: {n-test_start:,}. "
           f"Test negative-review rate: {y_test.mean():.2%}.\n\n"
           + result.to_markdown(index=False) + "\n")
(out / "three_approaches.md").write_text(summary)
print(summary)
import os
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(summary)
