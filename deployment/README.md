# Deployment

Use this folder for deployment-related artefacts required by the IT5006 project, such as application files, serving configuration, deployment instructions, or platform-specific assets.

Do not place exploratory notebooks here.

## Phase 1 Streamlit dashboard

The stakeholder-facing dashboard is available at `deployment/streamlit_app.py`. It
presents Phase 1 descriptive and exploratory views of orders, delivery performance,
customer experience, products, and geography. It does not include predictive modelling.

### Run locally

From the repository root:

```bash
pip install -r requirements.txt
streamlit run deployment/streamlit_app.py
```

The app reads only the prepared, one-row-per-order analytical dataset at
`data/processed/dashboard_orders.parquet`.

### Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app and select this repository.
3. Select branch `main` after the dashboard is merged.
4. Set the main file path to `deployment/streamlit_app.py`.
5. Deploy and copy the generated `.streamlit.app` URL.

Any future deployment should document:

- prerequisites
- configuration required
- how to run it
- expected inputs and outputs
- known limitations

Never commit real secrets or credentials. Use documented placeholders or environment variables instead.
