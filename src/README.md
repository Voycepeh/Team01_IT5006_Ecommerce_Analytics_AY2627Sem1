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

- `classification_data.py` constructs and validates the late-delivery negative-review model
  table from Phase 1 processed CSVs and owns the explicit feature contract.
- `classification_model.py` owns leakage-safe sklearn preprocessing, the dummy baseline,
  Logistic Regression, and the versioned inference artifact interface.
- `classification_evaluation.py` owns imbalance-aware metrics and training-only threshold
  comparison helpers.

See [`docs/phase2-late-order-classification-workflow.md`](../docs/phase2-late-order-classification-workflow.md)
for the big picture, component contracts, implementation decisions and parallel workflow.
