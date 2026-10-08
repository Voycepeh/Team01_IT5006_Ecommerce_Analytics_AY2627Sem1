# Phase 2 classification workflow

We investigate whether a delivered order will receive a negative customer review (1–2 stars). We make the prediction immediately after delivery, before the review is submitted.

![High-level Phase 2 classification workflow](assets/phase2-classification-workflow.svg)

## How we approach the problem

We start with the processed order and item tables from Phase 1. We build one row per order, filter out orders reviewed before delivery and prepare **21 candidate predictors** covering delivery performance, order value, order complexity, product characteristics, geography and purchase timing.

We separate earlier and later orders using an **80% chronological training / 20% test split**, then compare two model families: Logistic Regression and tree-based models (Decision Tree and Random Forest). We evaluate simple baselines before tuning. The five-fold time-series cross-validation runs **inside the training period** and selects a model using **PR AUC (average precision)** because negative reviews are the minority class.

## What we found

| Held-out metric | Tuned Logistic Regression | Tuned Random Forest |
|---|---:|---:|
| PR AUC | 0.187 | 0.188 |
| ROC AUC | 0.649 | 0.645 |
| Recall at threshold 0.50 | 2.3% | 0.1% |

We selected **tuned Random Forest during training cross-validation**, according to the pre-established PR AUC selection rule. On later held-out orders, however, the two tuned models have nearly identical ranking performance. We therefore **favour tuned Logistic Regression as a practical candidate** because it is simpler to interpret and implement. This is a pragmatic preference, not a retrospective change to our cross-validation selection criterion.

The 0.50 threshold detects very few negative reviews for either model. The notebook includes an **executed training-only forward-chaining out-of-fold threshold analysis** for both tuned finalists. It plots precision, recall and F1 across candidate thresholds, selects an illustrative operating point that maximises recall subject to at least 20% out-of-fold precision (falling back explicitly to best F1 if infeasible), and compares that frozen cutoff with 0.50 on the held-out period. The 20% precision floor is an analytical scenario, not an agreed stakeholder requirement. No threshold is selected using held-out outcomes. Threshold changes do not change PR AUC or ROC AUC. The threshold analysis has been executed using the committed processed Olist tables. Its observed outcomes are reported below; the 20% precision floor remains illustrative rather than a stakeholder-approved requirement.


### Threshold sensitivity on the chronological holdout

The executed threshold analysis shows why the default 0.50 cutoff is unsuitable for detecting most negative reviews. On the held-out period, Logistic Regression recall increased from **2.3% at 0.50** to **22.2% at a 0.12 threshold**, while precision decreased from 38.9% to 24.8%. Random Forest recall increased from **0.1% at 0.50** to **34.0% at 0.12**, with 18.6% precision. These cutoffs were selected from training out-of-fold predictions, not from the test labels. Lowering a cutoff changes the precision–recall trade-off, not the ranking quality (PR AUC or ROC AUC).

### Class imbalance sensitivity analysis

Only **8.46%** of the **18,235** orders in the held-out test period received a negative review. To investigate whether class imbalance was the primary explanation for weak detection, we compared three **Logistic Regression** training approaches: natural class distribution, balanced class weights, and 50/50 random undersampling. Each approach used the same chronological 64% fitting, 16% validation and 20% testing partitions. Thresholds were chosen to maximise F1 on the validation period; all three models were then refitted on the earlier 80% before one evaluation on the unchanged 20% holdout. Undersampling affected training data only.

| Logistic Regression training | PR AUC | ROC AUC | Validation-selected cutoff | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|
| Natural distribution | **0.1812** | 0.6440 | 0.13 | 23.17% | 22.75% | 0.2296 |
| Balanced class weights | 0.1793 | **0.6460** | 0.57 | 22.29% | **24.63%** | **0.2340** |
| 50/50 undersampling | 0.1778 | 0.6413 | 0.58 | **23.29%** | 22.49% | 0.2288 |

Balanced weights detected **380** negative reviews with **1,325** false alarms, compared with **351** detections and **1,164** false alarms using natural training. The 50/50 approach detected **347** negative reviews with **1,143** false alarms. None of the balancing approaches improved PR AUC over natural training, and the F1 differences were small. **We therefore retain the existing model training and selection workflow rather than adopting undersampling or switching the selected model based on this sensitivity check.** This indicates that rebalancing alone did not meaningfully resolve the model's limited predictive separation; it does not establish that imbalance has no effect.

These figures are a **separate Logistic Regression sensitivity experiment**, not a direct replacement for the earlier five-fold time-series cross-validation model comparison. The validation design and model-selection procedures differ, so their numbers should not be interpreted as directly interchangeable.

Reproducible source: [three-approach experiment script](../scripts/compare_phase2_class_balance.py). Recorded results: [CSV](../reports/phase2/three_approaches.csv). The threshold-analysis code is included in the main notebook; its saved GitHub copy does not currently embed the executed threshold plots or tables. The executed notebook was produced as a GitHub Actions artifact. Re-run the notebook to regenerate those outputs.

## Practical conclusion

Our cross-validation-selected Random Forest remains the formal selection result, while tuned Logistic Regression remains the simpler practical candidate given near-identical held-out ranking performance. Neither model has demonstrated sufficient negative-review detection at the default 0.50 cutoff for autonomous customer outreach. Exploratory threshold selection improved recall but increased false alarms. A real intervention policy would require stakeholder-defined costs, further validation and monitoring. The balancing experiment does not justify changing the main model families or training approach.

## Boundaries and reproducibility

Our predictors exclude review scores, review text and review timestamps. Historical review timestamps are used only for eligibility filtering, and delivery outcomes are available only because our prediction point is **after delivery**. Our chronological split is based on purchase date and does not fully reconstruct event-time label availability.

We implement and evaluate the workflow in [the simplified Phase 2 notebook](../notebooks/phase2_delivered_order_classification_simplified.ipynb). The original combined notebook remains available independently.

**Diagram note:** This figure uses the diagram-design editorial conventions: restrained accent, typographic hierarchy, grouped stages and orthogonal routing. It represents our notebook's scope rather than an operational deployment architecture.
