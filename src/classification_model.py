"""Leakage-safe sklearn components for negative-review classification."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.classification_data import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)


ARTIFACT_VERSION = 1
PREDICTION_POINT = "after customer delivery and before review submission"


def build_preprocessor() -> ColumnTransformer:
    """Build learned preprocessing that must be fitted inside each train fold."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES),
         ("categorical", categorical, CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def build_logistic_pipeline(
    *, random_state: int = 42, class_weight: str | dict[int, float] | None = None
) -> Pipeline:
    """Return an unfitted preprocessing + Logistic Regression pipeline."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=random_state,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def build_dummy_pipeline(*, strategy: str = "prior") -> Pipeline:
    """Return an unfitted baseline using the identical learned preprocessing."""
    return Pipeline(
        [("preprocessor", build_preprocessor()),
         ("classifier", DummyClassifier(strategy=strategy))]
    )


def save_inference_artifact(
    fitted_pipeline: Pipeline,
    path: str,
    *,
    threshold: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a deployment-ready bundle; callers must choose the threshold on training data."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not hasattr(fitted_pipeline, "classes_"):
        raise ValueError("fitted_pipeline must be fitted before it is saved")
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "pipeline": fitted_pipeline,
        "threshold": float(threshold),
        "model_features": MODEL_FEATURES,
        "prediction_point": PREDICTION_POINT,
        "metadata": metadata or {},
    }
    joblib.dump(artifact, path)


def predict_from_artifact(artifact: dict[str, Any], records: pd.DataFrame) -> pd.DataFrame:
    """Score contracted model rows and return probability plus thresholded class."""
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported classification artifact version")
    missing = set(MODEL_FEATURES).difference(records.columns)
    if missing:
        raise ValueError(f"Inference records are missing features: {sorted(missing)}")
    probabilities = artifact["pipeline"].predict_proba(records[MODEL_FEATURES])[:, 1]
    threshold = artifact["threshold"]
    return pd.DataFrame(
        {
            "negative_review_probability": probabilities,
            "predicted_negative_review": (probabilities >= threshold).astype("int8"),
        },
        index=records.index,
    )
