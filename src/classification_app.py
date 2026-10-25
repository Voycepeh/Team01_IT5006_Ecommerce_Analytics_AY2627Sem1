"""Validated order inputs and direct inference for the Phase 3 application."""
from pathlib import Path
import math
from numbers import Real
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

from src.classification_data import DELIVERED_MODEL_FEATURES, DELIVERED_NUMERIC_FEATURES
from src.classification_model import predict_from_delivered_order_artifact

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / 'deployment/classification/model.joblib'
COUNT_FEATURES = {'item_count', 'seller_count', 'payment_count', 'n_seller_states'}


def load_model(path=ARTIFACT_PATH):
    """Load only the repository-owned artifact; never accept uploaded pickle files."""
    artifact = joblib.load(path)
    classifier = artifact['pipeline'].named_steps['classifier']
    if (artifact.get('artifact_version') != 1
            or not isinstance(classifier, LogisticRegression)
            or list(classifier.classes_) != [0, 1]
            or artifact.get('model_scope') != 'all_delivered_orders'
            or artifact.get('model_features') != DELIVERED_MODEL_FEATURES
            or artifact.get('threshold') != .12
            or classifier.C != .01 or classifier.class_weight is not None):
        raise ValueError('Model does not match the final notebook contract')
    # Validate categorical preprocessing before the UI constructs its controls.
    category_options(artifact)
    return artifact


def category_options(artifact):
    preprocessor = artifact['pipeline'].named_steps['preprocessor']
    columns = preprocessor.transformers_[1][2]
    values = preprocessor.named_transformers_['categorical'].named_steps['one_hot'].categories_
    return {column: list(options) for column, options in zip(columns, values)}


def predict_order(artifact, values):
    """Validate one complete model row, then call the shared inference function."""
    missing = set(DELIVERED_MODEL_FEATURES) - values.keys()
    if missing:
        raise ValueError(f'Missing order information: {sorted(missing)}')
    for field in DELIVERED_NUMERIC_FEATURES:
        value = values[field]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f'{field} must be a finite number')
        if field != 'days_early' and value < 0:
            raise ValueError(f'{field} cannot be negative')
        if field in {'delivery_days', 'days_early'} and value != int(value):
            raise ValueError(f'{field} must be whole days')
        if field == 'max_installments' and value != int(value):
            raise ValueError('max_installments must be a non-negative whole number')
        if field in COUNT_FEATURES and (value < 1 or value != int(value)):
            raise ValueError(f'{field} must be a positive whole number')
    for field, options in category_options(artifact).items():
        if values[field] not in options:
            raise ValueError(f'Choose a supported value for {field}')
    if values['seller_count'] > values['item_count']:
        raise ValueError('Seller count cannot exceed item count')
    if values['n_seller_states'] > values['seller_count']:
        raise ValueError('Seller states cannot exceed seller count')
    if values['late_delivery_flag'] != int(values['days_early'] < 0):
        raise ValueError('Lateness must agree with days early')
    if values['delivery_days'] + values['days_early'] < 0:
        raise ValueError('Promised delivery cannot precede purchase')
    if not math.isfinite(values['order_item_value'] + values['order_freight_value']):
        raise ValueError('Combined item value and freight are too large')
    expected_share = (values['order_freight_value'] /
                      (values['order_item_value'] + values['order_freight_value'])
                      if values['order_item_value'] + values['order_freight_value'] else 0.)
    if not math.isclose(values['freight_share'], expected_share, abs_tol=1e-6):
        raise ValueError('Freight share must equal freight / (item value + freight)')
    if values['same_state'] != (values['customer_state'] == values['seller_state']):
        raise ValueError('Same-state indicator conflicts with customer and seller states')
    return predict_from_delivered_order_artifact(artifact, pd.DataFrame([values])).iloc[0]
