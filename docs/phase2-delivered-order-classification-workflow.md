# Phase 2 — Delivered-Order Negative Review Classification

## Question

For any completed delivered order, can Logistic Regression predict whether the
customer will subsequently submit a negative review of 1–2 stars?

- **Population:** delivered orders with an actual customer-delivery date and a
  subsequently recorded review.
- **Target:** `1` for review scores 1–2 and `0` for scores 3–5.
- **Prediction point:** immediately after customer delivery and before review
  submission.
- **Stakeholder use:** customer-experience teams can prioritise post-delivery
  service recovery across late, on-time and early orders.

This is a separate formulation from the late-order-only workflow. Their metrics
describe different populations and must not be treated as an apples-to-apples
model tournament.

## Population and class balance

The Phase 1 analytical data contains 96,470 delivered orders with an actual
delivery date. Of those, 95,824 (99.33%) have a recorded review and enter this
supervised dataset. There are 12,272 negative reviews, a target rate of 12.81%.

The class imbalance makes ordinary accuracy misleading. Evaluation therefore
emphasises precision, recall, F1, balanced accuracy, PR AUC and ROC AUC.

## Feature contract

The delivered-order model reuses the existing order, item, payment, product,
geography and timing features. It makes two deliberate changes:

1. `late_delivery_flag` becomes a predictor because both late and non-late orders
   are now in the population.
2. `estimated_delivery_days` is omitted because `delivery_days` and `days_early`
   already encode the same timing identity. Avoiding all three improves the
   interpretability of Logistic Regression coefficients.

Identifiers and all review-derived fields are excluded. Delivery outcomes are
valid because delivery has already occurred at the prediction point.

The authoritative lists are `DELIVERED_NUMERIC_FEATURES`,
`DELIVERED_CATEGORICAL_FEATURES` and `DELIVERED_MODEL_FEATURES` in
`src/classification_data.py`.

## Training and evaluation

The workflow uses an 80/20 stratified train-test split with random seed 42.
Numeric features receive median imputation and standardisation; categorical
features receive most-frequent imputation and one-hot encoding with unknown
categories ignored. All learned preprocessing remains inside the sklearn
Pipeline.

Five-fold stratified cross-validation on the training set reports F1 as a
stability check. The final unweighted Logistic Regression is fitted on all
training rows and evaluated once on the held-out test set using a fixed 0.50
threshold. No model tournament, hyperparameter search or threshold optimisation
is performed in this baseline notebook.

## Phase 3 handoff

The notebook saves the fitted pipeline and its explicit contract to:

`deployment/delivered_order_negative_review_pipeline.pkl`

The artifact is intentionally distinct from
`deployment/late_order_negative_review_pipeline.pkl`. It contains the fitted
preprocessing and classifier, threshold, delivered-order feature contract,
prediction point, model scope and metadata.
