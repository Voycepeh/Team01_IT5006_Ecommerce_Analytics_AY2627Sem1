from __future__ import annotations

import pandas as pd
import pytest

from src.classification_data import (
    DELIVERED_MODEL_FEATURES,
    DELIVERED_OUTPUT_COLUMNS,
    MODEL_FEATURES,
    OUTPUT_COLUMNS,
    TARGET_COLUMN,
    build_classification_dataset,
    build_delivered_order_classification_dataset,
    split_delivered_order_features_target,
    split_features_target,
)
from src.classification_model import (
    build_delivered_order_logistic_pipeline,
    build_logistic_pipeline,
    predict_from_artifact,
    predict_from_delivered_order_artifact,
)


def make_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    order_rows = []
    item_rows = []
    for index in range(12):
        order_id = f"order-{index}"
        order_rows.append(
            {
                "order_id": order_id,
                "order_status": "delivered",
                "order_delivered_customer_date": "2018-01-10",
                "review_answer_timestamp": "2018-01-12",
                "review_score": 1 if index % 3 == 0 else 5,
                "item_count": 1,
                "order_item_value": 20.0 + index,
                "order_freight_value": 4.0,
                "seller_count": 1,
                "payment_total": 24.0 + index,
                "payment_count": 1,
                "max_installments": 2,
                "freight_share": 0.15,
                "delivery_days": 5.0 + index % 2,
                "estimated_delivery_days": 10,
                # Phase 1 convention: negative means delivered late.
                "days_early": -1.0 - index % 2,
                "late_delivery_flag": 1.0,
                "n_seller_states": 1,
                "customer_state": "SP" if index % 2 else "RJ",
                "seller_state": "SP",
                "same_state": index % 2 == 1,
                "purchase_month": "2018-01",
                "purchase_weekday": "Monday",
                # A forbidden upstream column must not enter model output.
                "review_comment_message": "future information",
            }
        )
        item_rows.append(
            {
                "order_id": order_id,
                "order_item_id": 1,
                "price": 20.0 + index,
                "product_category_name_english": "health_beauty",
                "product_category_name": "beleza_saude",
                "product_weight_g": 500.0,
                "product_length_cm": 10.0,
                "product_height_cm": 5.0,
                "product_width_cm": 4.0,
                "product_photos_qty": 2.0,
            }
        )
    return pd.DataFrame(order_rows), pd.DataFrame(item_rows)


def test_dataset_has_explicit_schema_and_binary_target() -> None:
    orders, items = make_inputs()
    result = build_classification_dataset(orders, items)

    assert result.columns.tolist() == OUTPUT_COLUMNS
    assert result["order_id"].is_unique
    assert result[TARGET_COLUMN].tolist().count(1) == 4
    assert "review_comment_message" not in result
    assert result["product_volume_cm3_mean"].eq(200.0).all()
    features, _ = split_features_target(result)
    assert features.columns.tolist() == MODEL_FEATURES
    assert not any(column.startswith("review_") for column in features.columns)


def test_dataset_rejects_duplicate_order_grain() -> None:
    orders, items = make_inputs()
    orders = pd.concat([orders, orders.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate rows"):
        build_classification_dataset(orders, items)


@pytest.mark.parametrize("days_early", [0.0, 3.0])
def test_on_time_or_early_order_is_excluded_while_late_orders_remain(
    days_early: float,
) -> None:
    orders, items = make_inputs()
    excluded_order_id = orders.loc[0, "order_id"]
    orders.loc[0, "days_early"] = days_early
    orders.loc[0, "late_delivery_flag"] = 0.0

    result = build_classification_dataset(orders, items)

    assert excluded_order_id not in set(result["order_id"])
    assert len(result) == len(orders) - 1
    assert result["days_early"].lt(0).all()


def test_pipeline_and_inference_contract_work_end_to_end() -> None:
    orders, items = make_inputs()
    dataset = build_classification_dataset(orders, items)
    features, target = split_features_target(dataset)
    pipeline = build_logistic_pipeline(random_state=42)
    pipeline.fit(features, target)

    artifact = {
        "artifact_version": 1,
        "pipeline": pipeline,
        "threshold": 0.4,
        "model_features": MODEL_FEATURES,
    }
    predictions = predict_from_artifact(artifact, features.iloc[:2])

    assert predictions.shape == (2, 2)
    assert predictions["negative_review_probability"].between(0, 1).all()
    assert set(predictions["predicted_negative_review"]).issubset({0, 1})


def test_delivered_order_dataset_includes_late_and_non_late_orders() -> None:
    orders, items = make_inputs()
    orders.loc[:5, "days_early"] = 2.0
    orders.loc[:5, "late_delivery_flag"] = 0.0

    dataset = build_delivered_order_classification_dataset(orders, items)
    features, target = split_delivered_order_features_target(dataset)

    assert dataset.columns.tolist() == DELIVERED_OUTPUT_COLUMNS
    assert len(dataset) == len(orders)
    assert set(dataset["late_delivery_flag"]) == {0.0, 1.0}
    assert features.columns.tolist() == DELIVERED_MODEL_FEATURES
    assert "late_delivery_flag" in features
    assert "estimated_delivery_days" not in features
    assert target.tolist().count(1) == 4


def test_delivered_order_pipeline_and_inference_contract_work_end_to_end() -> None:
    orders, items = make_inputs()
    orders.loc[:5, "days_early"] = 2.0
    orders.loc[:5, "late_delivery_flag"] = 0.0
    dataset = build_delivered_order_classification_dataset(orders, items)
    features, target = split_delivered_order_features_target(dataset)
    pipeline = build_delivered_order_logistic_pipeline(random_state=42)
    pipeline.fit(features, target)

    artifact = {
        "artifact_version": 1,
        "model_scope": "all_delivered_orders",
        "pipeline": pipeline,
        "threshold": 0.5,
        "model_features": DELIVERED_MODEL_FEATURES,
    }
    predictions = predict_from_delivered_order_artifact(artifact, features.iloc[:2])

    assert predictions.shape == (2, 2)
    assert predictions["negative_review_probability"].between(0, 1).all()
    assert set(predictions["predicted_negative_review"]).issubset({0, 1})

