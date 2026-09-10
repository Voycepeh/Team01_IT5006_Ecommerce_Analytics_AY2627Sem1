# Deployment

Use this folder for deployment-related artefacts required by the IT5006 project, such as application files, serving configuration, deployment instructions, or platform-specific assets.

Do not place exploratory notebooks here.

## Phase 1 Streamlit dashboard

The stakeholder-facing dashboard is available at `deployment/streamlit/app.py`.
It is intentionally a higher-level companion to `notebooks/olist_eda.ipynb`, rather than a second independent analysis.

The dashboard focuses on three stakeholder views:

1. Business Overview: order volume and average order value over time.
2. Delivery Promise: actual delivery duration against Olist's checkout estimate.
3. Delivery & Customer Experience: review outcomes by delivery timing and same-state versus cross-state delivery patterns.

The default purchase-date window is January 2017 through August 2018, matching the stable temporal analysis window used in the EDA. Users can change the window for exploration.

The dashboard is descriptive only. It does not present Phase 2 predictive modelling or causal claims.

### Run locally

From the repository root:

```bash
pip install -r requirements.txt
python src/prepare_dashboard_data.py
streamlit run deployment/streamlit/app.py
```

The app reads the prepared, one-row-per-order analytical dataset at
`data/processed/dashboard_orders.parquet`.

### Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app and select this repository.
3. Select branch `main` after the dashboard is merged.
4. Set the main file path to `deployment/streamlit/app.py`.
5. Deploy and copy the generated `.streamlit.app` URL.

Any future deployment should document prerequisites, configuration, execution steps, expected inputs and outputs, and known limitations.

Never commit real secrets or credentials. Use documented placeholders or environment variables instead.
