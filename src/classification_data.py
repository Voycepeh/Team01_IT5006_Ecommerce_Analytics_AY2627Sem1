"""Construct the Phase 2 negative-review classification dataset.

This module consumes Phase 1 analytical CSV content, not the dashboard Parquet.
It deliberately stops at deterministic, business-specific feature construction;
learned preprocessing belongs in the fitted scikit-learn pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "is_negative_review"
IDENTIFIER_COLUMNS = ["order_id"]
NUMERIC_FEATURES = [
    "item_count",
    "order_item_value",
    "order_freight_value",
    "seller_count",
    "payment_total",
    "payment_count",
    "max_installments",
    "freight_share",
    "delivery_days",
    "estimated_delivery_days",
    "days_early",
    "n_seller_states",
    "product_weight_g_mean",
    "product_volume_cm3_mean",
    "product_photos_qty_mean",
]
CATEGORICAL_FEATURES = [
    "customer_state",
    "seller_state",
    "same_state",
    "purchase_month",
    "purchase_weekday",
    "primary_product_category",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + MODEL_FEATURES + [TARGET_COLUMN]

ORDER_REQUIRED_COLUMNS = {
    "order_id",
    "order_status",
    "order_delivered_customer_date",
    "review_score",
    "review_answer_timestamp",
    *NUMERIC_FEATURES[:-3],
    *CATEGORICAL_FEATURES[:-1],
}
ITEM_REQUIRED_COLUMNS = {
    "order_id",
    "order_item_id",
    "price",
    "product_category_name_english",
    "product_category_name",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "product_photos_qty",
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _validate_unique(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    if frame[columns].isna().any().any():
        raise ValueError(f"{name} has missing values in key {columns}")
    duplicates = int(frame.duplicated(columns).sum())
    if duplicates:
        raise ValueError(f"{name} has {duplicates:,} duplicate rows for key {columns}")


def _aggregate_item_features(item_level: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, order-grain product features known at purchase."""
    items = item_level.copy()
    items["_category"] = items["product_category_name_english"].fillna(
        items["product_category_name"]
    )
    items["_volume"] = (
        items["product_length_cm"]
        * items["product_height_cm"]
        * items["product_width_cm"]
    )

    summaries = (
        items.groupby("order_id", as_index=False)
        .agg(
            product_weight_g_mean=("product_weight_g", "mean"),
            product_volume_cm3_mean=("_volume", "mean"),
            product_photos_qty_mean=("product_photos_qty", "mean"),
        )
    )

    # Represent a multi-category order by the category with greatest item value.
    # Alphabetical sorting makes equal-value ties reproducible.
    category_value = (
        items.dropna(subset=["_category"])
        .groupby(["order_id", "_category"], as_index=False)["price"]
        .sum()
        .sort_values(
            ["order_id", "price", "_category"], ascending=[True, False, True]
        )
    )
    primary_category = (
        category_value.drop_duplicates("order_id")[["order_id", "_category"]]
        .rename(columns={"_category": "primary_product_category"})
    )
    result = summaries.merge(primary_category, on="order_id", how="left", validate="1:1")
    _validate_unique(result, ["order_id"], "classification item aggregates")
    return result


def build_classification_dataset(
    order_level: pd.DataFrame,
    item_level: pd.DataFrame,
) -> pd.DataFrame:
    """Build the model table for predicting a negative review after delivery.

    Required inputs
    ---------------
    ``order_level`` is the Phase 1 ``data/processed/order_level.csv`` content at
    exactly one row per ``order_id``. It must contain delivery, payment, order,
    customer/seller geography, temporal and deduplicated ``review_score`` fields.
    ``item_level`` is Phase 1 ``data/processed/item_level.csv`` content at one row
    per (``order_id``, ``order_item_id``), with price, category and product-size
    fields. The dashboard Parquet is intentionally not an accepted input.

    Prediction point and target
    ---------------------------
    The prediction is made immediately after a delivered order reaches the
    customer and before the customer submits a review. Eligible rows therefore
    have ``order_status == 'delivered'``, a customer-delivery timestamp, a
    deduplicated review score, and a review answer timestamp strictly after
    delivery. That temporal check ensures the stated prediction point exists.
    The binary target is 1 for scores 1--2 and 0 for scores 3--5. One output row
    represents one eligible order.

    Leakage restrictions
    --------------------
    Review identifiers, text and timestamps are never returned as predictors.
    Only the target is derived from review information. Identifiers are retained
    solely for traceability and are excluded by the explicit feature lists.
    Delivery outcomes are allowed because they exist at this prediction point;
    changing the prediction point requires a new reviewed feature contract.

    Output schema
    -------------
    Returns exactly ``order_id``, ``MODEL_FEATURES`` and
    ``is_negative_review``, in ``OUTPUT_COLUMNS`` order. Learned imputation,
    scaling and encoding are not performed here. Key uniqueness, source schema,
    item coverage, review-score domain and target validity are checked in code.
    """
    _require_columns(order_level, ORDER_REQUIRED_COLUMNS, "order_level")
    _require_columns(item_level, ITEM_REQUIRED_COLUMNS, "item_level")
    _validate_unique(order_level, ["order_id"], "order_level")
    _validate_unique(item_level, ["order_id", "order_item_id"], "item_level")

    non_null_scores = order_level["review_score"].dropna()
    invalid_scores = ~non_null_scores.isin([1, 2, 3, 4, 5])
    if invalid_scores.any():
        values = sorted(non_null_scores.loc[invalid_scores].unique().tolist())
        raise ValueError(f"review_score contains values outside 1--5: {values}")

    timestamps = order_level[
        ["order_delivered_customer_date", "review_answer_timestamp"]
    ].apply(pd.to_datetime, errors="coerce")
    malformed_dates = (
        order_level[["order_delivered_customer_date", "review_answer_timestamp"]]
        .notna()
        & timestamps.isna()
    )
    if malformed_dates.any().any():
        bad_columns = malformed_dates.any().loc[lambda values: values].index.tolist()
        raise ValueError(f"Unparseable timestamps in columns: {bad_columns}")

    eligible = order_level.loc[
        order_level["order_status"].eq("delivered")
        & timestamps["order_delivered_customer_date"].notna()
        & timestamps["review_answer_timestamp"].gt(
            timestamps["order_delivered_customer_date"]
        )
        & order_level["review_score"].notna()
    ].copy()
    if eligible.empty:
        raise ValueError("No delivered orders with review scores are available")

    item_features = _aggregate_item_features(item_level)
    missing_item_orders = ~eligible["order_id"].isin(item_features["order_id"])
    if missing_item_orders.any():
        raise ValueError(
            f"{int(missing_item_orders.sum()):,} eligible orders have no item rows"
        )

    dataset = eligible.merge(item_features, on="order_id", how="left", validate="1:1")
    dataset[TARGET_COLUMN] = dataset["review_score"].le(2).astype("int8")
    dataset["same_state"] = dataset["same_state"].astype("boolean")
    dataset = dataset[OUTPUT_COLUMNS].copy()

    _validate_unique(dataset, ["order_id"], "classification dataset")
    if not set(dataset[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise AssertionError("Classification target must be binary")
    if dataset[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Classification dataset must contain both target classes")
    if np.isinf(dataset[NUMERIC_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Numeric model features contain infinite values")
    return dataset.reset_index(drop=True)


def split_features_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return only the contracted model features and target."""
    _require_columns(dataset, set(OUTPUT_COLUMNS), "classification dataset")
    return dataset[MODEL_FEATURES].copy(), dataset[TARGET_COLUMN].copy()
