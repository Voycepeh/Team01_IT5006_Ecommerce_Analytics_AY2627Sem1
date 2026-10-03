"""Evaluation utilities that keep threshold selection away from the test set."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    """Calculate imbalance-aware metrics at one preselected threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    predictions = np.asarray(probabilities) >= threshold
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def threshold_table(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarise candidate thresholds using validation or OOF probabilities only."""
    candidates = thresholds if thresholds is not None else np.arange(0.05, 1.0, 0.05)
    return pd.DataFrame(
        [
            {"threshold": float(value), **classification_metrics(
                y_true, probabilities, threshold=float(value)
            )}
            for value in candidates
        ]
    )


def best_threshold_by_metric(table: pd.DataFrame, *, metric: str = "f1") -> float:
    """Return a statistical threshold candidate that maximises ``metric``.

    Ties favour the lower, recall-favouring threshold. This helper does not make
    the operating decision: the team must select that threshold from training
    or OOF analysis using the stakeholder costs of missed cases and unnecessary
    outreach. Protected test results must never be used for this choice.
    """
    required = {"threshold", metric}
    if missing := required.difference(table.columns):
        raise ValueError(f"threshold table is missing columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("threshold table must not be empty")
    if not pd.api.types.is_numeric_dtype(table[metric]):
        raise ValueError(f"metric must be numeric: {metric}")
    candidates = table.loc[np.isfinite(table[metric])]
    if candidates.empty:
        raise ValueError(f"metric has no finite values: {metric}")
    return float(
        candidates.sort_values([metric, "threshold"], ascending=[False, True])
        .iloc[0]["threshold"]
    )
