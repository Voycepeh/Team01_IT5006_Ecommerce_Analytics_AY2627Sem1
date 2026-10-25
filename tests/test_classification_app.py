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
    np.testing.assert_allclose(scored.negative_review_probability, expected, atol=1e-12, rtol=0)
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
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 1
    app.number_input[0].set_value(0).run()
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
