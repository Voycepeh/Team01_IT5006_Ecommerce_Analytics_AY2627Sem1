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

The 0.50 threshold detects very few negative reviews for either model. We do not choose a more favourable threshold using the held-out test set; a practical intervention threshold would require separate validation.

## Boundaries and reproducibility

Our predictors exclude review scores, review text and review timestamps. Historical review timestamps are used only for eligibility filtering, and delivery outcomes are available only because our prediction point is **after delivery**. Our chronological split is based on purchase date and does not fully reconstruct event-time label availability.

We implement and evaluate the workflow in [the simplified Phase 2 notebook](../notebooks/phase2_delivered_order_classification_simplified.ipynb). The original combined notebook remains available independently.

**Diagram note:** This figure uses the diagram-design editorial conventions: restrained accent, typographic hierarchy, grouped stages and orthogonal routing. It represents our notebook's scope rather than an operational deployment architecture.
