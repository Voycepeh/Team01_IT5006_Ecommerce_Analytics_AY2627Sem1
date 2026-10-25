"""Phase 3 contract, notebook parity, validation and Streamlit interaction checks."""
import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix
from streamlit.testing.v1 import AppTest

from src.classification_app import load_model, category_options, predict_order
from src.classification_data import DELIVERED_MODEL_FEATURES
from src.classification_model import predict_from_delivered_order_artifact

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'deployment/classification/app.py'


def test_export_matches_actual_final_notebook_cells():
    notebook = json.loads((ROOT / 'notebooks/phase2_delivered_order_classification_final.ipynb').read_text())
    namespace = {'__name__': '__notebook_parity__'}
    # Execute the authoritative population and split code, not an export-helper copy.
    for index in (2, 4):
        exec(compile(''.join(notebook['cells'][index]['source']), f'notebook-cell-{index}', 'exec'), namespace)
    model = namespace['build_delivered_order_logistic_pipeline'](random_state=42).set_params(
        classifier__C=.01, classifier__class_weight=None)
    model.fit(namespace['X_train'], namespace['y_train'])
    artifact = load_model()
    scored = predict_from_delivered_order_artifact(artifact, namespace['X_test'])
    expected = model.predict_proba(namespace['X_test'])[:, 1]
    # BLAS/platform rounding can differ around 1e-11. Keep decisions exact.
    np.testing.assert_array_equal(scored.predicted_negative_review, (expected >= .12).astype(int))
    np.testing.assert_allclose(scored.negative_review_probability, expected, atol=1e-9, rtol=0)
    assert confusion_matrix(namespace['y_test'], scored.predicted_negative_review).tolist() == [[15650, 1042], [1200, 343]]
    assert len(namespace['X_train']) == 72936


def valid_order():
    options = category_options(load_model())
    values = dict.fromkeys(DELIVERED_MODEL_FEATURES, 1.)
    for field, choices in options.items():
        values[field] = choices[0]
    values['customer_state'] = values['seller_state'] = 'SP'
    values.update(order_item_value=100., order_freight_value=20., freight_share=20/120,
                  delivery_days=10., days_early=5., late_delivery_flag=0,
                  same_state=values['customer_state'] == values['seller_state'])
    return values


@pytest.mark.parametrize('changes', [
    {'max_installments': -1}, {'max_installments': 1.5},
    {'order_item_value': 1e308, 'order_freight_value': 1e308},
    {'item_count': 0}, {'item_count': 1.5}, {'delivery_days': float('nan')},
    {'order_item_value': float('inf')}, {'seller_count': 2}, {'n_seller_states': 2},
    {'late_delivery_flag': 1}, {'freight_share': .8}, {'customer_state': 'invalid'},
    {'days_early': -20, 'late_delivery_flag': 1}, {'same_state': False},
])
def test_reject_invalid_orders(changes):
    values = valid_order()
    values.update(changes)
    with pytest.raises(ValueError):
        predict_order(load_model(), values)


def test_valid_prediction_and_missing_fields():
    artifact = load_model()
    values = valid_order()
    result = predict_order(artifact, values)
    assert 0 <= result.negative_review_probability <= 1
    assert result.predicted_negative_review == int(result.negative_review_probability >= .12)
    del values['item_count']
    with pytest.raises(ValueError, match='Missing'):
        predict_order(artifact, values)


def test_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / 'missing.joblib')


def test_streamlit_form():
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    app.button[0].click().run()
    assert 'Confirm' in app.error[0].value
    app.checkbox[0].check()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    assert len(app.metric) == 1
    app.number_input[0].set_value(0)
    app.button[0].click().run()
    assert app.error
    assert not app.metric  # No stale result after an invalid submission.


def test_streamlit_missing_artifact(monkeypatch):
    import src.classification_app as module
    def missing():
        raise FileNotFoundError('missing model')
    monkeypatch.setattr(module, 'load_model', missing)
    # Clear the app's resource cache so this really exercises the failure path.
    import streamlit as st
    st.cache_resource.clear()
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert 'could not be loaded' in app.error[0].value
    st.cache_resource.clear()


def test_numpy_numeric_inputs_and_zero_installments():
    values = valid_order()
    values['item_count'] = np.int64(1)
    values['max_installments'] = 0
    result = predict_order(load_model(), values)
    assert np.isfinite(result.negative_review_probability)


@pytest.mark.parametrize('field,value', [('artifact_version', 999), ('threshold', .5)])
def test_reject_wrong_artifact_contract(tmp_path, field, value):
    import joblib
    artifact = load_model()
    artifact[field] = value
    path = tmp_path / 'wrong.joblib'
    joblib.dump(artifact, path)
    with pytest.raises(ValueError, match='contract'):
        load_model(path)


def test_related_inputs_and_payment_override(monkeypatch):
    import src.classification_app as module
    original = module.predict_order
    captured = []

    def capture(artifact, values):
        captured.append(values.copy())
        return original(artifact, values)

    monkeypatch.setattr(module, 'predict_order', capture)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    app.checkbox(key='eligible').check()
    app.number_input(key='order_item_value').set_value(200.)
    app.number_input(key='order_freight_value').set_value(40.)
    app.number_input(key='days_early').set_value(-3)
    app.button[0].click().run()
    assert not app.error, [error.value for error in app.error]
    assert captured[-1]['payment_total'] == 240.
    assert captured[-1]['freight_share'] == pytest.approx(40/240)
    assert captured[-1]['late_delivery_flag'] == 1
    assert captured[-1]['same_state'] == (captured[-1]['customer_state'] == captured[-1]['seller_state'])
    assert set(captured[-1]) == set(DELIVERED_MODEL_FEATURES)

    app.number_input(key='payment_total').set_value(225.)
    app.button[0].click().run()
    assert not app.error
    assert captured[-1]['payment_total'] == 225.
    assert captured[-1]['freight_share'] == pytest.approx(40/240)
