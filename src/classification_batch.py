"""Batch CSV inference for the frozen delivered-order classification model."""
from __future__ import annotations

import io
import numpy as np
import pandas as pd

from src.classification_app import category_options
from src.classification_data import DELIVERED_MODEL_FEATURES, DELIVERED_NUMERIC_FEATURES
from src.classification_model import predict_from_delivered_order_artifact

REQUIRED = (
    "order_id", "item_count", "order_item_value", "order_freight_value",
    "delivery_days", "days_early", "customer_state", "seller_state",
)
DERIVED = ("freight_share", "late_delivery_flag", "same_state")
OPTIONAL = tuple(f for f in DELIVERED_MODEL_FEATURES if f not in REQUIRED and f not in DERIVED)
MAX_ROWS = 10000
TEMPLATE = pd.DataFrame([{
    "order_id": "EXAMPLE001", "item_count": 1, "order_item_value": 100,
    "order_freight_value": 20, "delivery_days": 10, "days_early": -2,
    "customer_state": "SP", "seller_state": "SP",
}])


def read_uploaded_csv(upload) -> pd.DataFrame:
    """Only read CSV, never executable/pickle uploads; retain identifiers as strings."""
    if hasattr(upload, "size") and upload.size > 10 * 1024 * 1024:
        raise ValueError("CSV exceeds the 10 MB limit")
    try:
        frame = pd.read_csv(upload, dtype={"order_id": "string"}, low_memory=False)
    except (pd.errors.ParserError, UnicodeError, ValueError, OSError) as exc:
        raise ValueError("Could not parse CSV. Save it as a UTF-8 comma-separated file.") from exc
    return frame


def prepare_batch(frame: pd.DataFrame, artifact):
    """Validate an order-grain input and derive features. Do not train or tune."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Expected CSV table")
    if frame.empty or len(frame) > MAX_ROWS:
        raise ValueError(f"Provide between 1 and {MAX_ROWS:,} order rows")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate CSV column names are not allowed")
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    records = frame.copy().reset_index(drop=True)
    ids = records["order_id"].astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("order_id must be present and unique for every row")
    for field in DERIVED:
        if field in records.columns:
            raise ValueError(f"Remove derived column {field}; the app calculates it")
    options = category_options(artifact)
    numeric = [f for f in DELIVERED_NUMERIC_FEATURES if f not in DERIVED]
    for field in numeric:
        if field not in records.columns:
            records[field] = np.nan
        else:
            raw = records[field]
            converted = pd.to_numeric(raw, errors="coerce")
            bad = raw.notna() & (raw.astype("string").str.strip() != "") & converted.isna()
            if bad.any():
                raise ValueError(f"{field}: nonnumeric value at CSV row {int(bad.idxmax()) + 2}")
            records[field] = converted
        value = records[field]
        if np.isinf(value.to_numpy(dtype=float)).any():
            raise ValueError(f"{field} must be finite")
        if field != "days_early" and value.lt(0).any():
            raise ValueError(f"{field} cannot be negative")
        if field in {"item_count", "seller_count", "payment_count", "n_seller_states"}:
            if value.dropna().mod(1).ne(0).any() or value.dropna().lt(1).any():
                raise ValueError(f"{field} must contain positive whole numbers")
        if field in {"delivery_days", "days_early", "max_installments"} and value.dropna().mod(1).ne(0).any():
            raise ValueError(f"{field} must contain whole numbers")
    for field, allowed in options.items():
        if field not in records.columns:
            records[field] = np.nan
        else:
            records[field] = records[field].replace(r"^\s*$", np.nan, regex=True)
            wrong = records[field].notna() & ~records[field].isin(allowed)
            if wrong.any():
                raise ValueError(f"{field}: unsupported category at CSV row {int(wrong.idxmax()) + 2}")
    for field in REQUIRED:
        if records[field].isna().any():
            raise ValueError(f"{field} is required for every row")
    if records["seller_count"].gt(records["item_count"]).any():
        raise ValueError("seller_count cannot exceed item_count")
    if records["n_seller_states"].gt(records["seller_count"]).any():
        raise ValueError("n_seller_states cannot exceed seller_count")
    if (records["delivery_days"] + records["days_early"]).lt(0).any():
        raise ValueError("Promised delivery cannot precede purchase")
    total = records["order_item_value"] + records["order_freight_value"]
    if np.isinf(total.to_numpy(dtype=float)).any():
        raise ValueError("Item and freight totals are too large")
    records["freight_share"] = (records["order_freight_value"] / total.replace(0, np.nan)).fillna(0)
    records["late_delivery_flag"] = records["days_early"].lt(0).astype(int)
    records["same_state"] = records["customer_state"].eq(records["seller_state"])
    # Optional features are genuinely missing: let the FROZEN pipeline impute
    # them using training-set statistics, never fabricated sample values.
    source_missing = [f for f in OPTIONAL if f not in frame.columns]
    missing_cells = int(records[list(OPTIONAL)].isna().sum().sum())
    return records[DELIVERED_MODEL_FEATURES], {
        "missing_optional_columns": source_missing,
        "missing_optional_cells": missing_cells,
    }


def score_batch(frame: pd.DataFrame, artifact):
    features, quality = prepare_batch(frame, artifact)
    scored = predict_from_delivered_order_artifact(artifact, features)
    result = frame.reset_index(drop=True).copy()
    result["negative_review_probability"] = scored["negative_review_probability"].round(6)
    result["follow_up_flag"] = scored["predicted_negative_review"].astype(bool)
    return result.sort_values("negative_review_probability", ascending=False, kind="stable"), quality


def demo_test_sample(root, *, n=100, seed=42):
    """Deterministic 100 rows from the actual chronologically held-out test set.

    Never expose review target in model inputs; read true labels separately only
    for optional retrospective demonstration, not threshold selection.
    """
    from src.export_classification import notebook_population
    population = notebook_population(root=root)
    cutoff = int(len(population) * 0.80)
    heldout = population.iloc[cutoff:]
    sampled = heldout.sample(n=n, random_state=seed).sort_values("purchase_time")
    labels = sampled.set_index("order_id")["is_negative_review"].astype(int)
    features = sampled[["order_id", *(f for f in DELIVERED_MODEL_FEATURES if f not in DERIVED)]].reset_index(drop=True)
    return features, labels
