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


def test_streamlit_batch_interface():
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any('Order Risk Assessment' in x.value for x in app.title)
    assert any('Assess negative-review risk' in b.label for b in app.button)


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


def test_batch_matches_single_order_and_optional_imputation():
    from src.classification_batch import TEMPLATE, prepare_batch, score_batch, REQUIRED
    artifact = load_model()
    values = valid_order()
    minimal = TEMPLATE.copy()
    minimal['order_id'] = ['ABC']
    minimal['days_early'] = [5]
    result, quality = score_batch(minimal, artifact)
    assert result['order_id'].iloc[0] == 'ABC'
    assert result['negative_review_probability'].between(0, 1).all()
    assert quality['missing_optional_cells'] > 0
    assert set(REQUIRED).issubset(minimal.columns)

    all_features = dict(values)
    all_features.pop('same_state')
    all_features.pop('freight_share')
    all_features.pop('late_delivery_flag')
    full = __import__('pandas').DataFrame([{'order_id': 'C1', **all_features}])
    scored, quality = score_batch(full, artifact)
    expected = predict_order(artifact, values)
    assert scored['negative_review_probability'].iloc[0] == pytest.approx(
        expected['negative_review_probability'], abs=1e-6)
    assert quality['missing_optional_cells'] == 0


@pytest.mark.parametrize('changes', [
    {'item_count': 0}, {'days_early': -20}, {'customer_state': 'invalid'},
    {'order_id': None}, {'order_item_value': -1},
])
def test_batch_rejects_invalid_rows(changes):
    from src.classification_batch import TEMPLATE, prepare_batch
    sample = TEMPLATE.copy()
    for field, value in changes.items():
        sample[field] = value
    with pytest.raises(ValueError):
        prepare_batch(sample, load_model())


def test_demo_is_only_heldout_and_reproducible():
    from src.classification_batch import demo_test_sample
    from src.export_classification import notebook_population
    population = notebook_population()
    heldout = population.iloc[int(len(population)*.8):]
    demo, labels = demo_test_sample(ROOT)
    assert len(demo) == 100
    assert demo['order_id'].is_unique
    assert set(demo['order_id']).issubset(set(heldout['order_id']))
    assert 'is_negative_review' not in demo.columns
    assert len(labels) == 100
    # Actual integration path: the held-out months can be unseen by training.
    from src.classification_batch import score_batch
    scored, quality = score_batch(demo, load_model())
    assert len(scored) == 100
    assert scored['negative_review_probability'].between(0, 1).all()


def test_unseen_optional_categorical_is_encoded_not_rejected():
    from src.classification_batch import TEMPLATE, score_batch
    data = TEMPLATE.copy()
    data['purchase_month'] = '2099-01'
    scored, _ = score_batch(data, load_model())
    assert len(scored) == 1
    assert scored['negative_review_probability'].between(0, 1).all()
