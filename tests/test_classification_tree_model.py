from __future__ import annotations

from pathlib import Path

import joblib

from src.classification_data import (
    DELIVERED_MODEL_FEATURES,
    build_delivered_order_classification_dataset,
    split_delivered_order_features_target,
)
from src.classification_model import (
    predict_from_delivered_order_artifact,
    save_delivered_order_inference_artifact,
)
from src.classification_tree_model import (
    build_delivered_order_decision_tree_pipeline,
    build_delivered_order_random_forest_pipeline,
)
from tests.test_classification_foundation import make_inputs


def make_delivered_features():
    orders, items = make_inputs()
    orders.loc[:5, "days_early"] = 2.0
    orders.loc[:5, "late_delivery_flag"] = 0.0
    dataset = build_delivered_order_classification_dataset(orders, items)
    return split_delivered_order_features_target(dataset)


def test_tree_preprocessing_does_not_scale_numeric_features() -> None:
    features, target = make_delivered_features()
    pipeline = build_delivered_order_decision_tree_pipeline(random_state=42)
    pipeline.fit(features, target)

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = list(preprocessor.get_feature_names_out())

    assert transformed[:, names.index("delivery_days")].tolist() == (
        features["delivery_days"].tolist()
    )


def test_random_forest_artifact_round_trip(tmp_path: Path) -> None:
    features, target = make_delivered_features()
    pipeline = build_delivered_order_random_forest_pipeline(
        random_state=42, n_estimators=10, n_jobs=1
    )
    pipeline.set_params(classifier__class_weight="balanced_subsample")
    pipeline.fit(features, target)

    path = tmp_path / "random_forest.pkl"
    save_delivered_order_inference_artifact(
        pipeline, str(path), threshold=0.5, compress=3
    )
    artifact = joblib.load(path)
    predictions = predict_from_delivered_order_artifact(artifact, features.iloc[:3])

    assert artifact["model_features"] == DELIVERED_MODEL_FEATURES
    assert predictions.shape == (3, 2)
    assert predictions["negative_review_probability"].between(0, 1).all()
    assert set(predictions["predicted_negative_review"]).issubset({0, 1})
