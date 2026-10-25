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

## Usage

1. Enter the common **Order details** and **Delivery and location** inputs. Days ahead of promise are positive for early delivery, zero for on-time delivery and negative for late delivery.
2. Expand **Additional details** when you need different payment information, product averages or historical purchase context. These are still model predictors, not optional features: the visible example defaults are used when you do not edit them. Purchase month and weekday start at the fitted training modes.
3. Leave total paid blank to calculate it from item total plus freight, or enter the recorded amount to override it. Freight share, lateness and same-state flags are calculated automatically.
4. Confirm the order is completed and has not yet been reviewed, then click **Assess review risk**. This is a human acknowledgement of intended use; review information and the confirmation are not predictors and the model cannot verify review status.
5. A score at or above 12% is flagged for prioritised follow-up. Below the cutoff does not mean no risk.

Freight share is freight divided by item value plus freight (zero when both are zero). Lateness and same-state flags are derived automatically. For multi-item orders, product weight, volume and photo count are arithmetic item means. Volume is length × height × width per item before averaging. Primary category is the category with greatest summed item price, with alphabetical tie breaking. Use the first seller state recorded in the team's order-level data.

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
