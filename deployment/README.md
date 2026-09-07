# Deployment

Use this folder for deployment-related artefacts required by the IT5006 project, such as application files, serving configuration, deployment instructions, or platform-specific assets.

Do not place exploratory notebooks here.

## Streamlit demo

A simple fake-data Streamlit app is available at `deployment/streamlit/app.py`. It exists only to test the GitHub → Streamlit Community Cloud deployment flow before the Phase 1 EDA dashboard is built.

### Run locally

From the repository root:

```bash
pip install -r requirements.txt
streamlit run deployment/streamlit/app.py
```

### Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app and select this repository.
3. Select branch `main` after this demo is merged.
4. Set the main file path to `deployment/streamlit/app.py`.
5. Deploy and copy the generated `.streamlit.app` URL.

The demo uses fake data only and requires no secrets, configuration, or Olist dataset files.

Any future deployment should document:

- prerequisites
- configuration required
- how to run it
- expected inputs and outputs
- known limitations

Never commit real secrets or credentials. Use documented placeholders or environment variables instead.
