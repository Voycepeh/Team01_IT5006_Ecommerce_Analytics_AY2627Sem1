"""Tree-based sklearn pipelines for the all-delivered negative-review model.

This is the second Phase 2 model family (Decision Tree / Random Forest). It uses
the same delivered-order feature contract as the Logistic Regression baseline, and
fitted pipelines are saved with ``save_delivered_order_inference_artifact`` so
Phase 3 can load either model with ``predict_from_delivered_order_artifact``.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.classification_data import (
    DELIVERED_CATEGORICAL_FEATURES,
    DELIVERED_NUMERIC_FEATURES,
)


def build_tree_preprocessor() -> ColumnTransformer:
    """Impute and one-hot encode the delivered-order features.

    Trees split on raw values ("delivery_days > 20?"), so numeric features are
    not scaled. Everything else matches the Logistic Regression preprocessing.
    """
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), DELIVERED_NUMERIC_FEATURES),
            ("categorical", categorical, DELIVERED_CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def _tree_pipeline(classifier) -> Pipeline:
    """Wrap a tree classifier; the ``classifier`` step name is what grid search tunes."""
    return Pipeline(
        [("preprocessor", build_tree_preprocessor()), ("classifier", classifier)]
    )


def build_delivered_order_decision_tree_pipeline(*, random_state: int = 42) -> Pipeline:
    """Return the simplest tree-family baseline: one unpruned Decision Tree."""
    return _tree_pipeline(DecisionTreeClassifier(random_state=random_state))


def build_delivered_order_random_forest_pipeline(
    *,
    random_state: int = 42,
    n_estimators: int = 200,
    n_jobs: int = -1,
) -> Pipeline:
    """Return an unfitted preprocessing + Random Forest pipeline.

    Other hyperparameters keep sklearn defaults so the notebook can tune them
    with ``GridSearchCV`` via ``classifier__<name>`` parameters.
    """
    return _tree_pipeline(
        RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    )
