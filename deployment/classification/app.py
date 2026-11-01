"""IT5006 Phase 3: batch post-delivery negative-review risk assessment."""
from pathlib import Path
import logging
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from src.classification_app import load_model
from src.classification_batch import (
    REQUIRED, OPTIONAL, TEMPLATE, demo_test_sample, read_uploaded_csv, score_batch,
)

st.set_page_config(page_title="Order Risk Assessment | IT5006", page_icon="📦", layout="wide")
st.title("Order Risk Assessment")
st.caption("Spot delivered orders that may need a customer follow-up, before a review is submitted.")
st.info("Historical Olist proof of concept. Predictions are for delivered, not-yet-reviewed orders.")

@st.cache_resource
def model():
    return load_model()

@st.cache_data(show_spinner="Preparing historical held-out orders…")
def historical_demo():
    return demo_test_sample(ROOT)

try:
    artifact = model()
except Exception:
    logging.exception("Could not load classification artifact")
    st.error("The classification model could not be loaded. No prediction was made.")
    st.stop()

with st.sidebar:
    st.header("About & Help")
    with st.expander("How to use", expanded=True):
        st.write("1. Upload your CSV or load 100 historical demo orders.")
        st.write("2. Run the classification model for all orders.")
        st.write("3. Review and download the scored CSV.")
        st.caption("Each CSV row is one delivered order, after delivery and before review. Review scores are never model inputs.")
    with st.expander("CSV requirements"):
        st.write("Required columns:")
        st.code(", ".join(REQUIRED), language=None)
        st.write("Optional predictor columns:")
        st.code(", ".join(OPTIONAL), language=None)
        st.caption("Missing optional values use the trained pipeline's imputation. The app derives freight share, late-delivery flag and same-state automatically.")
        st.download_button("Download CSV template", TEMPLATE.to_csv(index=False), "order_template.csv", "text/csv")
    with st.expander("How it works"):
        st.write("Validate each row → derive known features → impute missing optional values using training statistics → preprocess and run fitted Logistic Regression → return risk probability.")
        st.write("A risk score of at least 12% is flagged for possible customer follow-up. This threshold was frozen from the final notebook.")
    with st.expander("Model and limitations"):
        st.write("Tuned Logistic Regression (C=0.01), frozen 12% threshold. Historical held-out precision: 24.8%; recall: 22.2%.")
        st.write("Scores are estimates, not calibrated guarantees. Most historical flags were false alarms. Missing predictors may make scores less reliable; the reduced CSV schema has not been independently evaluated.")
        st.write("Historical 2016–2018 Olist data only. No automatic customer intervention or live database connectivity.")

st.subheader("1. Select orders")
upload_tab, demo_tab = st.tabs(["Upload CSV", "Load Demo Data"])
with upload_tab:
    upload = st.file_uploader("Upload completed orders", type=["csv"], help="One unique order per row; CSV up to 10 MB")
    st.caption("Future feature: Connect to backend order database.")
with demo_tab:
    st.write("Try the app with 100 reproducibly sampled orders from the chronological held-out test set. These are historical records, not new production orders.")
    load_demo = st.button("Load 100 demo orders", use_container_width=False)

if load_demo:
    try:
        demo_frame, demo_labels = historical_demo()
        st.session_state["source_frame"] = demo_frame
        st.session_state["demo_labels"] = demo_labels
        st.session_state["source_name"] = "Historical held-out demo"
        st.session_state.pop("scored", None)
        st.session_state.pop("quality", None)
    except Exception:
        logging.exception("Failed to prepare held-out demo")
        st.error("Could not load the historical demo. The training/test source files may be unavailable.")
if upload is not None:
    upload_signature = (upload.name, upload.size, upload.getvalue().__hash__())
    if st.session_state.get("upload_signature") != upload_signature:
        try:
            st.session_state["source_frame"] = read_uploaded_csv(upload)
            st.session_state["source_name"] = upload.name
            st.session_state["upload_signature"] = upload_signature
            st.session_state.pop("demo_labels", None)
            st.session_state.pop("scored", None)
            st.session_state.pop("quality", None)
        except ValueError as exc:
            st.error(str(exc))

frame = st.session_state.get("source_frame")
if frame is not None:
    st.success(f"{st.session_state['source_name']}: {len(frame):,} orders selected")
    with st.expander("Preview input rows"):
        st.dataframe(frame.head(10), use_container_width=True, hide_index=True)

st.subheader("2. Run risk assessment")
if st.button("Assess negative-review risk", type="primary", disabled=frame is None):
    try:
        result, quality = score_batch(frame, artifact)
        st.session_state["scored"] = result
        st.session_state["quality"] = quality
    except ValueError as exc:
        st.session_state.pop("scored", None)
        st.session_state.pop("quality", None)
        st.error(f"CSV validation: {exc}")
    except Exception:
        logging.exception("Batch scoring failed")
        st.session_state.pop("scored", None)
        st.session_state.pop("quality", None)
        st.error("The batch could not be scored. No results were generated.")

result = st.session_state.get("scored")
if result is not None:
    st.subheader("3. Review and export")
    flagged = int(result["follow_up_flag"].sum())
    a,b,c = st.columns(3)
    a.metric("Orders scored", f"{len(result):,}")
    b.metric("Flagged for follow-up", f"{flagged:,}")
    c.metric("Share flagged", f"{flagged/len(result):.1%}")
    q = st.session_state["quality"]
    if q["missing_optional_cells"]:
        st.warning(
            f"{q['missing_optional_cells']:,} optional feature values were missing "
            "and handled by the fitted model's training-derived imputers. "
            "Reduced-input prediction quality has not been independently validated."
        )
    st.dataframe(
        result, use_container_width=True, hide_index=True,
        column_config={
            "negative_review_probability": st.column_config.NumberColumn(
                "Negative review probability", format="%.1f%%",
                help="Probability is stored as 0–1 in the downloaded CSV.",
            ),
            "follow_up_flag": st.column_config.CheckboxColumn("Follow up (≥12%)"),
        },
    )
    st.download_button(
        "Download scored CSV", data=result.to_csv(index=False),
        file_name="orders_with_review_risk.csv", mime="text/csv", type="primary",
    )
    if "demo_labels" in st.session_state:
        with st.expander("Historical demo: compare against actual outcomes"):
            labels = st.session_state["demo_labels"]
            evaluation = result[["order_id", "negative_review_probability", "follow_up_flag"]].copy()
            evaluation["actual_negative_review"] = evaluation["order_id"].map(labels)
            st.caption("Retrospective comparison only. True review outcomes were not used for scoring or adjusting the threshold; this 100-order sample is not a new model evaluation.")
            st.dataframe(evaluation, hide_index=True, use_container_width=True)
