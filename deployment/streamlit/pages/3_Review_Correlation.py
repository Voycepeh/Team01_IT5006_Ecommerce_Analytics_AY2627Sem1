"""Stakeholder-friendly review correlation view derived from the Phase 1 EDA."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "processed" / "dashboard_orders.parquet"
BLUE = "#2563EB"
RED = "#DC2626"
GREEN = "#0F766E"
GREY = "#64748B"

st.set_page_config(page_title="Review Correlations", page_icon="🔎", layout="wide")


@st.cache_data(show_spinner="Loading dashboard data…")
def load_data(path: str) -> pd.DataFrame:
    """Load the prepared one-row-per-order dashboard dataset."""
    frame = pd.read_parquet(path).copy()
    frame["purchase_date"] = pd.to_datetime(frame.get("purchase_date"), errors="coerce")
    return frame


def review_correlation_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson correlations against review score using EDA-aligned order features."""
    data = frame.copy()

    if "days_late" in data.columns:
        data["days_early"] = -pd.to_numeric(data["days_late"], errors="coerce")
    if {"customer_state", "seller_state"}.issubset(data.columns):
        has_route = data["customer_state"].notna() & data["seller_state"].notna()
        data["same_state"] = pd.NA
        data.loc[has_route, "same_state"] = data.loc[has_route, "customer_state"].eq(
            data.loc[has_route, "seller_state"]
        ).astype(float)

    candidates = {
        "delivery_days": "Actual delivery days",
        "estimated_delivery_days": "Promised delivery days",
        "days_early": "Days early vs promise",
        "days_to_review_request": "Days to review request",
        "total_item_value": "Order item value",
        "order_item_value": "Order item value",
        "total_freight_value": "Freight value",
        "order_freight_value": "Freight value",
        "freight_share": "Freight share",
        "item_count": "Item count",
        "seller_count": "Seller count",
        "max_installments": "Maximum installments",
        "same_state": "Customer and seller in same state",
    }

    rows = []
    for column, label in candidates.items():
        if column not in data.columns:
            continue
        pair = pd.DataFrame(
            {
                "feature": pd.to_numeric(data[column], errors="coerce"),
                "review_score": pd.to_numeric(data["review_score"], errors="coerce"),
            }
        ).dropna()
        if len(pair) < 3 or pair["feature"].nunique() < 2 or pair["review_score"].nunique() < 2:
            continue
        rows.append(
            {
                "Feature": label,
                "Pearson r": pair["feature"].corr(pair["review_score"], method="pearson"),
                "Orders": len(pair),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["Absolute r"] = result["Pearson r"].abs()
    return result.sort_values("Absolute r", ascending=True).drop(columns="Absolute r")


def correlation_figure(correlations: pd.DataFrame) -> go.Figure:
    """Show the review-score correlations as a compact stakeholder-facing bar chart."""
    colors = [RED if value < 0 else GREEN for value in correlations["Pearson r"]]
    fig = go.Figure(
        go.Bar(
            x=correlations["Pearson r"],
            y=correlations["Feature"],
            orientation="h",
            marker_color=colors,
            text=correlations["Pearson r"],
            texttemplate="%{text:+.2f}",
            textposition="outside",
            customdata=correlations[["Orders"]],
            hovertemplate="%{y}<br>Pearson r: %{x:+.2f}<br>Orders: %{customdata[0]:,.0f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=GREY, line_width=1)
    fig.update_layout(
        template="plotly_white",
        height=max(420, 46 * len(correlations) + 120),
        title="What factors are associated with review score?",
        xaxis={"title": "Pearson correlation with review score", "range": [-1, 1], "tickformat": ".1f"},
        yaxis={"title": ""},
        margin={"l": 20, "r": 70, "t": 70, "b": 45},
        showlegend=False,
    )
    return fig


def main() -> None:
    """Render the review-correlation dashboard page."""
    st.title("🔎 Review Correlation Analysis")
    st.caption(
        "A compact dashboard view of the Pearson correlation analysis already used in the EDA notebook. "
        "Negative review means review score below 3 (scores 1 or 2)."
    )

    try:
        data = load_data(str(DATA_PATH))
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Unable to load the dashboard dataset: {exc}")
        st.stop()

    if "review_score" not in data.columns:
        st.error("The prepared dashboard dataset does not contain review_score.")
        st.stop()

    if "purchase_date" in data.columns and data["purchase_date"].notna().any():
        minimum = data["purchase_date"].min().date()
        maximum = data["purchase_date"].max().date()
        date_range = st.sidebar.date_input(
            "Purchase date",
            value=(minimum, maximum),
            min_value=minimum,
            max_value=maximum,
        )
        if len(date_range) == 2:
            start = pd.Timestamp(date_range[0])
            end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            data = data[data["purchase_date"].between(start, end)].copy()

    if "is_delivered_complete" in data.columns:
        data = data[data["is_delivered_complete"].fillna(False)].copy()
    data = data.dropna(subset=["review_score"]).copy()

    if data.empty:
        st.info("No reviewed delivered orders match the selected period.")
        return

    correlations = review_correlation_data(data)
    if correlations.empty:
        st.info("There are not enough numeric review-related features to calculate correlations for the current selection.")
        return

    strongest = correlations.iloc[correlations["Pearson r"].abs().argmax()]
    direction = "negative" if strongest["Pearson r"] < 0 else "positive"
    st.info(
        f"**What stands out:** The strongest displayed linear association with review score is "
        f"{strongest['Feature']} (r = {strongest['Pearson r']:+.2f}), a {direction} relationship. "
        "This is exploratory association, not evidence of causation."
    )

    st.plotly_chart(correlation_figure(correlations), use_container_width=True)
    st.caption(
        "Pearson r ranges from -1 to +1. Values near 0 indicate little linear association; "
        "values farther from 0 indicate stronger linear association. Correlation does not imply causation."
    )

    with st.expander("See correlation values"):
        display = correlations.sort_values("Pearson r")[["Feature", "Pearson r", "Orders"]].copy()
        display["Pearson r"] = display["Pearson r"].map(lambda value: f"{value:+.3f}")
        st.dataframe(display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
