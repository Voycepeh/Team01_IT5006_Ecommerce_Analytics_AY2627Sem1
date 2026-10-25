"""Reproduce the final notebook's selected training fit; never fit on test rows."""
from pathlib import Path
import hashlib
import json

import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score

from src.classification_data import (
    build_delivered_order_classification_dataset, split_delivered_order_features_target,
)
from src.classification_model import (
    build_delivered_order_logistic_pipeline, save_delivered_order_inference_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'deployment/classification'
NOTEBOOK = ROOT / 'notebooks/phase2_delivered_order_classification_final.ipynb'


def notebook_population(root=ROOT):
    """Mirror final notebook cells 2 and 4, including stable chronological ordering."""
    orders = pd.read_csv(root / 'data/processed/order_level.csv')
    items = pd.read_csv(root / 'data/processed/item_level.csv')
    dataset = build_delivered_order_classification_dataset(orders, items)
    lookup = orders.set_index('order_id').loc[dataset['order_id']]
    dataset['purchase_time'] = pd.to_datetime(
        lookup['order_purchase_timestamp'], format='mixed').to_numpy()
    reviewed = pd.to_datetime(lookup['review_answer_timestamp'], format='mixed').to_numpy()
    delivered = pd.to_datetime(lookup['order_delivered_customer_date'], format='mixed').to_numpy()
    return dataset.loc[~(reviewed < delivered)].sort_values(
        'purchase_time', kind='stable').reset_index(drop=True)


def export():
    dataset = notebook_population()
    X, y = split_delivered_order_features_target(dataset)
    cut = int(len(dataset) * .80)
    pipeline = build_delivered_order_logistic_pipeline(random_state=42).set_params(
        classifier__C=.01, classifier__class_weight=None)
    pipeline.fit(X.iloc[:cut], y.iloc[:cut])
    probabilities = pipeline.predict_proba(X.iloc[cut:])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y.iloc[cut:], probabilities >= .12).ravel()
    # Exact counts published in the authoritative notebook, not new model selection.
    expected = [15650, 1042, 1200, 343]
    if [int(tn), int(fp), int(fn), int(tp)] != expected:
        raise ValueError('Reproduced predictions disagree with final notebook counts')
    metadata = {
        'notebook': str(NOTEBOOK.relative_to(ROOT)),
        'notebook_sha256': hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        'sklearn_version': sklearn.__version__,
        'train_rows': cut, 'test_rows': len(dataset) - cut,
        'parameters': {'C': .01, 'class_weight': None, 'random_state': 42},
        'test_confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'test_average_precision': average_precision_score(y.iloc[cut:], probabilities),
        'test_roc_auc': roc_auc_score(y.iloc[cut:], probabilities),
        'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ['data/processed/order_level.csv', 'data/processed/item_level.csv']},
    }
    OUTPUT.mkdir(exist_ok=True)
    save_delivered_order_inference_artifact(pipeline, str(OUTPUT / 'model.joblib'),
                                           threshold=.12, metadata=metadata, compress=3)
    (OUTPUT / 'model_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    export()
