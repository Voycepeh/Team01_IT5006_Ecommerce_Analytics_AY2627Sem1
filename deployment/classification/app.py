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

st.set_page_config(page_title="Customer Experience Risk Monitor | IT5006", page_icon="📦", layout="wide")
st.title("Customer Experience Risk Monitor")
st.markdown("**Identify at-risk customers before they leave a negative review.**")
st.write("Prioritise customer follow-ups using predicted review risk, so your team can focus attention where it matters most.")

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
        st.markdown("**Required columns**")
        for column in REQUIRED:
            st.markdown("* " + column)
        st.markdown("**Optional predictor columns**")
        for column in OPTIONAL:
            st.markdown("* " + column)
        st.caption("Missing optional values use the fitted model's training-derived imputation. Freight share, lateness and matching states are automatically calculated.")
        st.download_button("Download CSV template", TEMPLATE.to_csv(index=False), "order_template.csv", "text/csv")
    with st.expander("How it works"):
        st.markdown(
            "* **Upload your CSV**, or connect to an order database (**future feature**).\n"
            "* **Check required columns** for complete, valid inputs.\n"
            "* **Fill missing optional values** from training statistics.\n"
            "* **Prepare the data** for prediction.\n"
            "* **Run the trained Logistic Regression model** to assess review risk.\n"
            "* **Return a risk score** for each order.\n"
            "* **Recommend follow-up** when the risk score is **12% or higher**."
        )
    with st.expander("Model and limitations"):
        st.markdown(
            "* **Model:** Tuned Logistic Regression (C=0.01), fixed 12% follow-up cutoff.\n"
            "* **Historical test metrics:** Precision 24.8%; recall 22.2%.\n"
            "* **Training data:** 72,936 earlier eligible Olist orders from 2016–2018; "
            "18,235 later orders reserved for testing.\n"
            "* **Input quality:** Where possible, supply all 21 original predictors. "
            "The published test metrics do not establish equivalent performance for reduced-input CSVs."
        )
        st.markdown("**Limitations**")
        st.markdown(
            "* **Imbalanced reviews:** Only 8.5% of test orders received 1–2 stars.\n"
            "* **False positives:** 1,042 of 1,385 flagged test orders did not receive a negative "
            "review; 343 did. The 12% cutoff was chosen with a validation precision floor of 20%, "
            "but many negative reviews were still missed.\n"
            "* **Human judgement:** Customer Experience staff should review flagged orders before "
            "deciding on outreach; a flag is not proof of dissatisfaction.\n"
            "* **Uncertainty:** Scores are not calibrated guarantees. Optional missing inputs "
            "may reduce reliability, and reduced-schema accuracy has not been independently evaluated.\n"
            "* **Scope:** Historical Olist records; no live backend or automatic customer outreach."
        )

st.subheader("1. Select orders")
upload_tab, demo_tab = st.tabs(["Upload CSV", "Explore Historical Orders"])
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

st.subheader("2. Identify at-risk orders")
if st.button("Identify at-risk orders", type="primary", disabled=frame is None):
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
    st.subheader("3. Prioritise and export follow-ups")
    flagged = int(result["follow_up_flag"].sum())
    a,b,c = st.columns(3)
    a.metric("Orders scored", f"{len(result):,}")
    b.metric("Follow-ups recommended", f"{flagged:,}")
    c.metric("Share flagged", f"{flagged/len(result):.1%}")
    q = st.session_state["quality"]
    if q["missing_optional_cells"]:
        st.warning(
            f"{q['missing_optional_cells']:,} optional feature values were missing "
            "and handled by the fitted model's training-derived imputers. "
            "Reduced-input prediction quality has not been independently validated."
        )
    display = result.copy()
    display["Negative Review Risk (%)"] = (display["negative_review_probability"] * 100).round(1)
    display["Follow-up Recommended"] = display["follow_up_flag"].map({True: "Yes", False: "No"})
    display = display.drop(columns=["negative_review_probability", "follow_up_flag"])
    first = ["order_id", "Negative Review Risk (%)", "Follow-up Recommended"]
    display = display[first + [col for col in display if col not in first]]
    display = display.rename(columns={"order_id": "Order ID"})

    def highlight_results(row):
        flagged_row = row["Follow-up Recommended"] == "Yes"
        return [
            "background-color: #dcecf5; color: #14354d; font-weight: bold"
            if col in ("Negative Review Risk (%)", "Follow-up Recommended")
            else "background-color: #fff0d9; color: #5b3c17" if flagged_row else ""
            for col in row.index
        ]

    st.dataframe(
        display.style.apply(highlight_results, axis=1),
        use_container_width=True, hide_index=True,
        column_config={
            "Order ID": st.column_config.TextColumn("Order ID", width="medium"),
            "Negative Review Risk (%)": st.column_config.NumberColumn("Negative Review Risk (%)", format="%.1f"),
            "Follow-up Recommended": st.column_config.TextColumn("Follow-up Recommended", width="medium"),
        },
    )
    st.download_button(
        "Export follow-up list", data=result.to_csv(index=False),
        file_name="orders_with_review_risk.csv", mime="text/csv", type="primary",
    )
