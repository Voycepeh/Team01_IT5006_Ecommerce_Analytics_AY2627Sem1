"""Phase 1 stakeholder dashboard aligned to the Olist exploratory analysis."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "dashboard_orders.parquet"
)

ANALYSIS_START = pd.Timestamp("2017-01-01")
ANALYSIS_END = pd.Timestamp("2018-08-31")

PAGES = (
    "Business Overview",
    "Delivery Promise",
    "Delivery & Customer Experience",
)

BLUE = "#2563EB"
ORANGE = "#F97316"
GREEN = "#0F766E"
RED = "#DC2626"
GREY = "#64748B"

st.set_page_config(
    page_title="Olist E-commerce Analytics",
    page_icon="📦",
    layout="wide",
)


@st.cache_data(show_spinner="Loading dashboard data…")
def load_data(path: str) -> pd.DataFrame:
    """Load the validated one-row-per-order analytical dataset."""
    frame = pd.read_parquet(path)
    required = {
        "order_id",
        "customer_unique_id",
        "purchase_date",
        "order_status",
        "customer_state",
        "seller_state",
        "product_category",
        "total_item_value",
        "review_score",
        "delivery_days",
        "estimated_delivery_days",
        "days_late",
        "is_delivered_complete",
        "is_late",
        "is_negative_review",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Dashboard dataset is missing required columns: {sorted(missing)}"
        )

    frame = frame.copy()
    frame["purchase_date"] = pd.to_datetime(frame["purchase_date"], errors="coerce")
    return frame


def options(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def apply_filters(
    frame: pd.DataFrame,
    date_range,
    customer_states,
    seller_states,
    categories,
    statuses,
) -> pd.DataFrame:
    """Apply dashboard filters with AND logic."""
    result = frame
    if len(date_range) == 2:
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        result = result[result["purchase_date"].between(start, end)]

    for column, selected in (
        ("customer_state", customer_states),
        ("seller_state", seller_states),
        ("product_category", categories),
        ("order_status", statuses),
    ):
        if selected:
            result = result[result[column].isin(selected)]

    return result.copy()


def money(value: float) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1_000_000:
        return f"R${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"R${value / 1_000:,.1f}K"
    return f"R${value:,.2f}"


def number(value: float, decimals: int = 1) -> str:
    return "—" if pd.isna(value) else f"{value:,.{decimals}f}"


def percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def styled(figure: go.Figure, height: int = 390) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
        legend_title_text="",
        hoverlabel={"namelength": -1},
    )
    return figure


def chart(figure: go.Figure, key: str) -> None:
    st.plotly_chart(figure, use_container_width=True, key=key)


def delivered_orders(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["is_delivered_complete"]
        & frame["delivery_days"].notna()
        & frame["estimated_delivery_days"].notna()
        & frame["days_late"].notna()
        & frame["is_late"].notna()
    ].copy()


def business_overview(frame: pd.DataFrame) -> None:
    st.header("Business Overview")
    st.caption(
        "High-level context for the EDA: how order volume and basket value changed over the selected period."
    )

    delivered = delivered_orders(frame)
    reviewed = frame["review_score"].dropna()

    cols = st.columns(5)
    cols[0].metric("Orders", f"{frame['order_id'].nunique():,}")
    cols[1].metric(
        "Unique customers", f"{frame['customer_unique_id'].nunique():,}"
    )
    cols[2].metric(
        "Gross merchandise value",
        money(frame["total_item_value"].sum(min_count=1)),
        help="Sum of item prices, excluding freight. This is GMV, not Olist revenue.",
    )
    cols[3].metric("Average order value", money(frame["total_item_value"].mean()))
    cols[4].metric("Average review score", number(reviewed.mean(), 2))

    monthly = (
        frame.assign(Month=frame["purchase_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=False)
        .agg(
            Orders=("order_id", "nunique"),
            AOV=("total_item_value", "mean"),
        )
        .sort_values("Month")
    )

    if monthly.empty:
        st.info("No monthly observations match the current filters.")
        return

    indexed = monthly.dropna(subset=["Orders", "AOV"]).copy()
    if not indexed.empty and indexed.iloc[0]["Orders"] > 0 and indexed.iloc[0]["AOV"] > 0:
        indexed["Order volume index"] = indexed["Orders"] / indexed.iloc[0]["Orders"] * 100
        indexed["AOV index"] = indexed["AOV"] / indexed.iloc[0]["AOV"] * 100
        indexed_long = indexed.melt(
            id_vars="Month",
            value_vars=["Order volume index", "AOV index"],
            var_name="Measure",
            value_name="Index",
        )
        fig = px.line(
            indexed_long,
            x="Month",
            y="Index",
            color="Measure",
            markers=True,
            title="Order volume and average order value, indexed to first selected month = 100",
            color_discrete_map={
                "Order volume index": BLUE,
                "AOV index": ORANGE,
            },
        )
        fig.add_hline(y=100, line_dash="dash", line_color=GREY)
        chart(styled(fig, 430), "overview-indexed-growth")

    left, right = st.columns(2)
    with left:
        fig = px.line(
            monthly,
            x="Month",
            y="Orders",
            markers=True,
            title="Monthly order volume",
            color_discrete_sequence=[BLUE],
        )
        chart(styled(fig), "overview-orders")

    with right:
        fig = px.line(
            monthly,
            x="Month",
            y="AOV",
            markers=True,
            title="Monthly average order value",
            color_discrete_sequence=[ORANGE],
        )
        fig.update_yaxes(tickprefix="R$")
        chart(styled(fig), "overview-aov")

    if not delivered.empty:
        st.caption(
            f"Delivery metrics use completed delivered orders with valid delivery timestamps. "
            f"{delivered['order_id'].nunique():,} orders are eligible under the current filters."
        )


def delivery_promise(frame: pd.DataFrame) -> None:
    st.header("Delivery Promise")
    st.caption(
        "How conservative is the delivery estimate shown to customers at checkout?"
    )

    valid = delivered_orders(frame)
    if valid.empty:
        st.info("No delivered orders with valid quote and delivery timestamps match the filters.")
        return

    valid["days_early"] = -valid["days_late"].astype(float)
    quote_mae = valid["days_early"].abs().mean()

    cols = st.columns(4)
    cols[0].metric("Median actual delivery", f"{number(valid['delivery_days'].median(), 0)} days")
    cols[1].metric("Median quoted delivery", f"{number(valid['estimated_delivery_days'].median(), 0)} days")
    cols[2].metric(
        "Median arrival vs promise",
        f"{number(valid['days_early'].median(), 0)} days early",
        help="Positive means the order arrived before the estimated delivery date.",
    )
    cols[3].metric(
        "Current quote MAE",
        f"{number(quote_mae, 1)} days",
        help="Mean absolute error of the existing Olist delivery promise against actual delivery.",
    )

    left, right = st.columns(2)

    with left:
        quote = (
            valid.groupby("estimated_delivery_days", as_index=False)
            .agg(
                **{
                    "Median actual delivery days": ("delivery_days", "median"),
                    "Orders": ("order_id", "nunique"),
                }
            )
        )
        fig = px.scatter(
            quote,
            x="estimated_delivery_days",
            y="Median actual delivery days",
            size="Orders",
            labels={"estimated_delivery_days": "Quoted delivery days"},
            title="Quoted versus actual delivery duration",
            color_discrete_sequence=[ORANGE],
        )
        axis_max = max(
            float(quote["estimated_delivery_days"].max()),
            float(quote["Median actual delivery days"].max()),
        )
        fig.add_trace(
            go.Scatter(
                x=[0, axis_max],
                y=[0, axis_max],
                mode="lines",
                name="Actual = quoted",
                line={"color": GREY, "dash": "dash"},
            )
        )
        chart(styled(fig, 430), "promise-quoted-actual")

    with right:
        lower = float(valid["days_early"].quantile(0.01))
        upper = float(valid["days_early"].quantile(0.99))
        trimmed = valid[valid["days_early"].between(lower, upper)]
        fig = px.histogram(
            trimmed,
            x="days_early",
            nbins=45,
            title="How early or late orders arrive relative to the promise",
            labels={"days_early": "Days early (negative = late)"},
            color_discrete_sequence=[BLUE],
        )
        fig.add_vline(x=0, line_dash="dash", line_color=RED)
        fig.update_yaxes(title="Orders")
        chart(styled(fig, 430), "promise-error-distribution")

    st.caption(
        "The dashboard describes the incumbent quote only. It does not claim a better model yet; "
        "that is a Phase 2 regression question."
    )


def delivery_experience(frame: pd.DataFrame) -> None:
    st.header("Delivery & Customer Experience")
    st.caption(
        "How is delivery timing associated with review outcomes, and how does geography relate to delivery performance?"
    )

    valid = delivered_orders(frame)
    reviewed = valid.dropna(subset=["review_score", "is_negative_review"]).copy()

    if reviewed.empty:
        st.info("No reviewed orders with valid delivery outcomes match the filters.")
        return

    reviewed["days_early"] = -reviewed["days_late"].astype(float)
    bins = [-float("inf"), -30, -15, -7, -3, 0, 3, 7, 15, float("inf")]
    labels = [
        "30+ late",
        "15–30 late",
        "7–15 late",
        "3–7 late",
        "0–3 late",
        "0–3 early",
        "3–7 early",
        "7–15 early",
        "15+ early",
    ]
    reviewed["Delivery timing"] = pd.cut(
        reviewed["days_early"],
        bins=bins,
        labels=labels,
        ordered=True,
    )

    timing = (
        reviewed.dropna(subset=["Delivery timing"])
        .groupby("Delivery timing", observed=False, as_index=False)
        .agg(
            **{
                "Average review score": ("review_score", "mean"),
                "Negative review rate": ("is_negative_review", "mean"),
                "Reviewed orders": ("order_id", "nunique"),
            }
        )
    )
    display_order = list(reversed(labels))
    timing["Delivery timing"] = pd.Categorical(
        timing["Delivery timing"].astype(str),
        categories=display_order,
        ordered=True,
    )
    timing = timing.sort_values("Delivery timing")

    cols = st.columns(3)
    cols[0].metric("Reviewed delivered orders", f"{reviewed['order_id'].nunique():,}")
    cols[1].metric("Average review score", number(reviewed["review_score"].mean(), 2))
    cols[2].metric("Negative review rate", percent(reviewed["is_negative_review"].mean()))

    fig = px.bar(
        timing,
        x="Delivery timing",
        y="Average review score",
        hover_data={
            "Negative review rate": ":.1%",
            "Reviewed orders": True,
        },
        title="Review score by delivery timing relative to the promise",
        color="Average review score",
        color_continuous_scale="RdYlGn",
    )
    fig.update_yaxes(range=[1, 5])
    fig.update_layout(coloraxis_showscale=False)
    chart(styled(fig, 430), "experience-timing")

    geography = valid.dropna(subset=["customer_state", "seller_state"]).copy()
    if not geography.empty:
        geography["Route"] = geography.apply(
            lambda row: "Same state"
            if row["customer_state"] == row["seller_state"]
            else "Cross state",
            axis=1,
        )
        route = (
            geography.groupby("Route", as_index=False)
            .agg(
                **{
                    "Median delivery days": ("delivery_days", "median"),
                    "Late delivery rate": ("is_late", "mean"),
                    "Eligible orders": ("order_id", "nunique"),
                }
            )
        )

        left, right = st.columns(2)
        with left:
            fig = px.bar(
                route,
                x="Route",
                y="Median delivery days",
                hover_data=["Eligible orders"],
                title="Median delivery time by route type",
                color="Route",
                color_discrete_map={"Same state": GREEN, "Cross state": ORANGE},
            )
            chart(styled(fig), "experience-route-days")

        with right:
            fig = px.bar(
                route,
                x="Route",
                y="Late delivery rate",
                hover_data=["Eligible orders"],
                title="Late delivery rate by route type",
                color="Route",
                color_discrete_map={"Same state": GREEN, "Cross state": RED},
            )
            fig.update_yaxes(tickformat=".0%")
            chart(styled(fig), "experience-route-late")

    st.caption(
        "These are descriptive associations, not causal estimates. Seller state is the dashboard's "
        "dominant seller state for an order, so multi-seller orders are simplified for this view."
    )


def main() -> None:
    st.title("📦 Olist E-commerce Analytics")
    st.caption(
        "Phase 1 stakeholder dashboard supporting the exploratory analysis and later problem scoping"
    )

    try:
        data = load_data(str(DATA_PATH))
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Unable to load the dashboard dataset: {exc}")
        st.stop()

    minimum = data["purchase_date"].min().date()
    maximum = data["purchase_date"].max().date()
    default_start = max(ANALYSIS_START.date(), minimum)
    default_end = min(ANALYSIS_END.date(), maximum)

    st.sidebar.header("Dashboard")
    page = st.sidebar.radio("View", PAGES)

    st.sidebar.header("Filters")
    date_range = st.sidebar.date_input(
        "Purchase date range",
        value=(default_start, default_end),
        min_value=minimum,
        max_value=maximum,
        help="The default window matches the stable temporal analysis period used in the EDA: Jan 2017 to Aug 2018.",
    )
    customer_states = st.sidebar.multiselect(
        "Customer state", options(data["customer_state"])
    )
    seller_states = st.sidebar.multiselect(
        "Primary seller state", options(data["seller_state"])
    )
    categories = st.sidebar.multiselect(
        "Primary product category", options(data["product_category"])
    )
    statuses = st.sidebar.multiselect("Order status", options(data["order_status"]))

    filtered = apply_filters(
        data,
        date_range,
        customer_states,
        seller_states,
        categories,
        statuses,
    )

    if len(date_range) == 2:
        st.info(
            f"Analysis window: {pd.Timestamp(date_range[0]).strftime('%b %Y')} to "
            f"{pd.Timestamp(date_range[1]).strftime('%b %Y')} · "
            f"{filtered['order_id'].nunique():,} orders selected"
        )

    if filtered.empty:
        st.warning("No orders match the current filters.")
        return

    {
        "Business Overview": business_overview,
        "Delivery Promise": delivery_promise,
        "Delivery & Customer Experience": delivery_experience,
    }[page](filtered)


if __name__ == "__main__":
    main()
