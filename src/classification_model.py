"""Sklearn components for late-only and all-delivered review classification."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.classification_data import (
    CATEGORICAL_FEATURES,
    DELIVERED_CATEGORICAL_FEATURES,
    DELIVERED_MODEL_FEATURES,
    DELIVERED_NUMERIC_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
)

ARTIFACT_VERSION = 1
PREDICTION_POINT = "after a late customer delivery and before review submission"
DELIVERED_ORDER_PREDICTION_POINT = (
    "after customer delivery and before review submission"
)


def _build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build learned preprocessing for an explicit feature contract."""
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
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        verbose_feature_names_out=False,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build late-order preprocessing fitted inside each train fold."""
    return _build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)


def build_logistic_pipeline(*, random_state: int = 42) -> Pipeline:
    """Return an unfitted preprocessing + Logistic Regression pipeline."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_delivered_order_logistic_pipeline(*, random_state: int = 42) -> Pipeline:
    """Return the all-delivered preprocessing + Logistic Regression pipeline."""
    preprocessor = _build_preprocessor(
        DELIVERED_NUMERIC_FEATURES,
        DELIVERED_CATEGORICAL_FEATURES,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=random_state),
            ),
        ]
    )


def save_inference_artifact(
    fitted_pipeline: Pipeline,
    path: str,
    *,
    threshold: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save the fitted preprocessing + Logistic Regression pipeline for inference."""
    _save_inference_artifact(
        fitted_pipeline,
        path,
        threshold=threshold,
        model_features=MODEL_FEATURES,
        prediction_point=PREDICTION_POINT,
        metadata=metadata,
    )


def _save_inference_artifact(
    fitted_pipeline: Pipeline,
    path: str,
    *,
    threshold: float,
    model_features: list[str],
    prediction_point: str,
    metadata: dict[str, Any] | None,
    model_scope: str | None = None,
    compress: int = 0,
) -> None:
    """Persist one fitted pipeline with its explicit inference contract.

    ``compress`` (0-9) is passed to joblib; large tree ensembles use it to stay
    small enough for GitHub. ``joblib.load`` reads either form unchanged.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not hasattr(fitted_pipeline, "classes_"):
        raise ValueError("fitted_pipeline must be fitted before it is saved")
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "pipeline": fitted_pipeline,
        "threshold": float(threshold),
        "model_features": model_features,
        "prediction_point": prediction_point,
        "metadata": metadata or {},
    }
    if model_scope is not None:
        artifact["model_scope"] = model_scope
    joblib.dump(artifact, path, compress=compress)


def predict_from_artifact(artifact: dict[str, Any], records: pd.DataFrame) -> pd.DataFrame:
    """Score contracted model rows and return probability plus thresholded class."""
    return _predict_from_artifact(artifact, records, model_features=MODEL_FEATURES)


def _predict_from_artifact(
    artifact: dict[str, Any],
    records: pd.DataFrame,
    *,
    model_features: list[str],
    expected_scope: str | None = None,
) -> pd.DataFrame:
    """Validate and score records against one explicit model contract."""
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported classification artifact version")
    if expected_scope is not None and artifact.get("model_scope") != expected_scope:
        raise ValueError(f"Artifact is not a {expected_scope} model")
    missing = set(model_features).difference(records.columns)
    if missing:
        raise ValueError(f"Inference records are missing features: {sorted(missing)}")
    probabilities = artifact["pipeline"].predict_proba(records[model_features])[:, 1]
    threshold = artifact["threshold"]
    return pd.DataFrame(
        {
            "negative_review_probability": probabilities,
            "predicted_negative_review": (probabilities >= threshold).astype("int8"),
        },
        index=records.index,
    )


def save_delivered_order_inference_artifact(
    fitted_pipeline: Pipeline,
    path: str,
    *,
    threshold: float,
    metadata: dict[str, Any] | None = None,
    compress: int = 0,
) -> None:
    """Save the fitted all-delivered pipeline and its inference contract."""
    _save_inference_artifact(
        fitted_pipeline,
        path,
        threshold=threshold,
        model_features=DELIVERED_MODEL_FEATURES,
        prediction_point=DELIVERED_ORDER_PREDICTION_POINT,
        metadata=metadata,
        model_scope="all_delivered_orders",
        compress=compress,
    )


def predict_from_delivered_order_artifact(
    artifact: dict[str, Any], records: pd.DataFrame
) -> pd.DataFrame:
    """Score rows using the all-delivered artifact contract."""
    return _predict_from_artifact(
        artifact,
        records,
        model_features=DELIVERED_MODEL_FEATURES,
        expected_scope="all_delivered_orders",
    )
