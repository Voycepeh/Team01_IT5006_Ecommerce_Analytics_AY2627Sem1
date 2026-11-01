# Phase 3: delivered-order negative-review classification

Standalone Streamlit POC for customer service. Predicts a 1–2 star review **after delivery and before review submission**. The Phase 1 dashboard and regression work are separate and unchanged. No API server or FastAPI is used.

[Open the live classification POC](https://team01it5006ecommerceanalyticsay2627sem1-tsrzb29fqhr2k5khdoplo.streamlit.app/)

## Run

Use Python 3.12 from the repository root:

```bash
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r deployment/classification/requirements.txt
python -m streamlit run deployment/classification/app.py
```

The app needs only its committed `model.joblib`, `src/` modules and the dedicated requirements. It does not load raw/processed data, fit a model, or download anything at runtime. Load only repository-owned joblib files: pickle-based formats execute Python code and must never be accepted as user uploads.

## Three-step app workflow

1. **Select orders:** upload a UTF-8 CSV (at most 10 MB, up to 10,000 unique orders), or click **Load 100 demo orders**. The demo deterministically samples the chronological 20% held-out test population, excluding the original training rows. It reads the committed Phase 1 processed datasets when requested, never retrains the model, and never passes review outcomes to inference. Future backend/database integration is a labelled placeholder only.
2. **Run risk assessment:** validate input, derive freight share, lateness and state match, and score using the existing frozen Logistic Regression pipeline. Required CSV fields: `order_id, item_count, order_item_value, order_freight_value, delivery_days, days_early, customer_state, seller_state`. Every other model predictor is optional and may be imputed by the fitted training-time preprocessing; uploaders are warned when this happens. A template is available directly in the app sidebar. The source data should only contain delivered, not-yet-reviewed orders; this is a documented assumption, not independently verified by the model.
3. **Review and export:** inspect probability percentages and flags based on the frozen 12% threshold. Export the original CSV data with `negative_review_probability` (0–1) and `follow_up_flag` (boolean) appended. The demo additionally allows an explicitly labelled retrospective outcome comparison.

The in-app **About & Help** sidebar contains usage guidance, CSV requirements, an explanation of preprocessing, historical model performance and limitations. The complete-feature trained model's held-out results do **not** establish equivalent performance for reduced-input CSV uploads. Missing predictor fields can materially affect accuracy.

The cutoff is an illustrative validation policy, not a business-approved intervention rule. The score is not a calibrated guarantee. Outreach effects have not been established. Historical test precision is 24.8%, recall 22.2%; about three quarters of flags were false alarms. Do not automatically penalise customers or sellers.

## Reproduce the artifact

Authoritative analysis: `notebooks/phase2_delivered_order_classification_final.ipynb`. Older `deployment/*.pkl` files are not used or replaced.

```bash
python -m src.export_classification
python -m pip install pytest matplotlib ipython
python -m pytest tests/test_classification_app.py -q
```

Export requires the committed Phase 1 `data/processed/order_level.csv` and `item_level.csv`. It reuses `classification_data.py` and `classification_model.py`, excludes reviews submitted before delivery, stably sorts by purchase timestamp, fits only the earliest 72,936 orders (80%), and leaves 18,235 later orders for evaluation. It fits the already selected Logistic Regression with `C=0.01`, `class_weight=None`, seed 42, and the existing imputation/scaling/one-hot pipeline. No tuning or test-driven cutoff changes occur. The final notebook does not save a fitted Python object, so this is a reproducible refit of that selected training estimator, not extraction of a persisted notebook object.

The export fails if its held-out confusion counts disagree with the notebook. Metadata records source and notebook hashes, fit parameters and held-out metrics. The integration test independently executes the notebook's actual preparation/split cells, fits its selected estimator and compares **every held-out probability** to the exported pipeline within 1e-9 (allowing machine-level numerical differences), with every threshold decision required to match exactly. It also checks the recorded matrix `[[15650,1042],[1200,343]]`. This does not rerun the tree searches or repeat model selection.

## Streamlit Community Cloud

Create a **new app**, retaining the existing Phase 1 deployment:

* Repository: `Voycepeh/Team01_IT5006_Ecommerce_Analytics_AY2627Sem1`
* Branch: this PR branch for preview; `main` after human review and merge
* Main file: `deployment/classification/app.py`
* Advanced settings: Python **3.12**
* Dependencies: `requirements.txt` beside this entrypoint pins its independent runtime
* Secrets: none

See [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app). After creating the app, verify its live URL opens, submit a valid order, reject an ineligible order and an invalid count, and compare the live result with the same local inputs. Record that verified URL in the final report and capture screenshots for the locked presentation deck. The artifact and app can be included in the final deployment ZIP along with source and requirements.

**Status: the original POC was deployed and live smoke-tested on 25 October 2026.** A valid example returned 6.1% risk, matching the local score 0.06134775282714478; unconfirmed order status and zero item count were rejected. The revised layout must be checked live after its PR is merged and Community Cloud updates. No claim is made that the revised layout is already live.

## Academic review

ChatGPT/Codex assisted with the app, export, tests and documentation. Team members must review and understand the implementation and declare AI assistance in assessed submissions. This implements only classification deployment; the project's regression requirements remain owned by the regression workstream.
