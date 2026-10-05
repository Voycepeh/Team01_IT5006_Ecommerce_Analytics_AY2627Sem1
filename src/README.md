# Source Code

Use this folder for reusable project code that supports data preparation, analysis, modelling, evaluation, or deployment.

## Conventions

- Prefer small, focused modules with clear responsibilities.
- Add docstrings or comments where behaviour is not obvious.
- Avoid embedding secrets or machine-specific paths.
- Keep functions reusable where practical.
- Update `requirements.txt` when introducing new third-party packages.
- Add checks or tests when a change could materially affect project results.

## Phase 2 classification

- `classification_data.py` constructs and validates both the late-delivery and
  all-delivered negative-review model tables from Phase 1 processed CSVs and owns
  their explicit feature contracts.
- `classification_model.py` owns leakage-safe sklearn preprocessing, Logistic Regression,
  and the distinct versioned inference artifact interfaces.
- `classification_evaluation.py` owns imbalance-aware metrics and training-only threshold
  comparison helpers.
- `classification_tree_model.py` owns the second model family (Decision Tree / Random
  Forest) for the all-delivered problem. Its pipelines reuse the delivered-order feature
  contract and are saved with the same artifact interface as Logistic Regression.

See [`docs/phase2-late-order-classification-workflow.md`](../docs/phase2-late-order-classification-workflow.md)
for the original late-order workflow and
[`docs/phase2-delivered-order-classification-workflow.md`](../docs/phase2-delivered-order-classification-workflow.md)
for the expanded delivered-order workflow.
