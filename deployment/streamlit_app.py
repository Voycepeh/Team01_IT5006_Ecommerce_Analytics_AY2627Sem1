"""Phase 1 stakeholder dashboard for the Olist e-commerce dataset."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "dashboard_orders.parquet"
)
PAGES = (
    "Overview",
    "Delivery Performance",
    "Customer Experience",
    "Product & Geography",
)
BLUE, ORANGE, GREEN, RED, GREY = "#2563EB", "#F97316", "#0F766E", "#DC2626", "#64748B"

st.set_page_config(
    page_title="Olist E-commerce Performance", page_icon="📦", layout="wide"
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
        "total_payment_value",
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
    frame, date_range, customer_states, seller_states, categories, statuses
):
    """Apply all global filters with AND logic."""
    result = frame
    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
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


def styled(figure: go.Figure, height: int = 380) -> go.Figure:
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


def order_counts(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    return (
        frame.dropna(subset=[column])
        .groupby(column, observed=True)["order_id"]
        .nunique()
        .rename("Orders")
        .reset_index()
        .rename(columns={column: label})
    )


def rates(frame, dimension, flag, label, minimum_orders=1):
    eligible = frame.dropna(subset=[dimension, flag])
    if eligible.empty:
        return pd.DataFrame(columns=[label, "Rate", "Eligible orders"])
    result = (
        eligible.groupby(dimension, observed=True)
        .agg(Rate=(flag, "mean"), **{"Eligible orders": ("order_id", "nunique")})
        .reset_index()
        .rename(columns={dimension: label})
    )
    return result[result["Eligible orders"] >= minimum_orders]


def overview(frame: pd.DataFrame) -> None:
    st.header("Overview")
    st.caption("What is happening in the Olist e-commerce business?")
    reviewed, lateness = frame["review_score"].dropna(), frame["is_late"].dropna()
    cols = st.columns(6)
    cols[0].metric("Orders", f"{frame['order_id'].nunique():,}")
    cols[1].metric("Unique customers", f"{frame['customer_unique_id'].nunique():,}")
    cols[2].metric(
        "Total item value", money(frame["total_item_value"].sum(min_count=1))
    )
    cols[3].metric("Avg. order value", money(frame["total_item_value"].mean()))
    cols[4].metric("Avg. review score", number(reviewed.mean(), 2))
    cols[5].metric("Late-delivery rate", percent(lateness.mean()))

    left, right = st.columns(2)
    with left:
        monthly = (
            frame.assign(
                Month=frame["purchase_date"].dt.to_period("M").dt.to_timestamp()
            )
            .groupby("Month", as_index=False)
            .agg(Orders=("order_id", "nunique"))
        )
        fig = px.line(
            monthly, x="Month", y="Orders", markers=True, title="Monthly order trend"
        )
        fig.update_traces(line_color=BLUE)
        chart(styled(fig), "overview-orders")
    with right:
        monthly_value = (
            frame.dropna(subset=["total_payment_value"])
            .assign(
                Month=lambda x: x["purchase_date"].dt.to_period("M").dt.to_timestamp()
            )
            .groupby("Month", as_index=False)["total_payment_value"]
            .sum()
            .rename(columns={"total_payment_value": "Payment value"})
        )
        if monthly_value.empty:
            st.info("No payment values match the current filters.")
        else:
            fig = px.area(
                monthly_value,
                x="Month",
                y="Payment value",
                title="Monthly payment value",
            )
            fig.update_traces(line_color=GREEN, fillcolor="rgba(15,118,110,0.22)")
            fig.update_yaxes(tickprefix="R$", tickformat="~s")
            chart(styled(fig), "overview-payments")

    left, right = st.columns(2)
    with left:
        review_dist = (
            frame.dropna(subset=["review_score"])
            .groupby("review_score", as_index=False)["order_id"]
            .nunique()
            .rename(columns={"review_score": "Review score", "order_id": "Orders"})
        )
        if review_dist.empty:
            st.info("No reviews match the current filters.")
        else:
            fig = px.bar(
                review_dist,
                x="Review score",
                y="Orders",
                title="Review score distribution",
                color_discrete_sequence=[BLUE],
            )
            fig.update_xaxes(dtick=1)
            chart(styled(fig), "overview-reviews")
    with right:
        category = order_counts(frame, "product_category", "Primary category").nlargest(
            10, "Orders"
        )
        if category.empty:
            st.info("No primary product categories match the current filters.")
        else:
            fig = px.bar(
                category.sort_values("Orders"),
                x="Orders",
                y="Primary category",
                orientation="h",
                title="Top primary product categories by orders",
                color_discrete_sequence=[ORANGE],
            )
            chart(styled(fig), "overview-categories")

    states = order_counts(frame, "customer_state", "Customer state").sort_values(
        "Orders", ascending=False
    )
    fig = px.bar(
        states,
        x="Customer state",
        y="Orders",
        title="Customer orders by state",
        color="Orders",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False)
    chart(styled(fig), "overview-states")


def delivery(frame: pd.DataFrame) -> None:
    st.header("Delivery Performance")
    st.caption(
        "How long do deliveries take, and where does Olist miss its promised date?"
    )
    st.info(
        "Days late compares actual delivery with the estimated date: negative values mean early; "
        "positive values mean after the estimate. Missing delivery/lateness outcomes are excluded."
    )
    valid = frame[
        frame["is_delivered_complete"]
        & frame["delivery_days"].notna()
        & frame["days_late"].notna()
        & frame["is_late"].notna()
    ].copy()
    cols = st.columns(4)
    cols[0].metric("Eligible delivered orders", f"{valid['order_id'].nunique():,}")
    cols[1].metric("Median delivery days", number(valid["delivery_days"].median()))
    cols[2].metric(
        "Median quoted days", number(valid["estimated_delivery_days"].median())
    )
    cols[3].metric("Late-order rate", percent(valid["is_late"].mean()))
    if valid.empty:
        st.info("No orders with valid delivery outcomes match the current filters.")
        return

    left, right = st.columns(2)
    with left:
        upper = max(float(valid["delivery_days"].quantile(0.99)), 1)
        fig = px.histogram(
            valid[valid["delivery_days"] <= upper],
            x="delivery_days",
            nbins=35,
            title="Delivery duration distribution (up to 99th percentile)",
            labels={"delivery_days": "Delivery days"},
            color_discrete_sequence=[BLUE],
        )
        fig.update_yaxes(title="Orders")
        chart(styled(fig), "delivery-duration")
    with right:
        quote = valid.groupby("estimated_delivery_days", as_index=False).agg(
            **{
                "Median actual delivery days": ("delivery_days", "median"),
                "Orders": ("order_id", "nunique"),
            }
        )
        fig = px.scatter(
            quote,
            x="estimated_delivery_days",
            y="Median actual delivery days",
            size="Orders",
            labels={"estimated_delivery_days": "Quoted delivery days"},
            title="Estimated vs actual delivery duration",
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
        chart(styled(fig), "delivery-quote")

    left, right = st.columns(2)
    with left:
        monthly = (
            valid.assign(
                Month=valid["purchase_date"].dt.to_period("M").dt.to_timestamp()
            )
            .groupby("Month", as_index=False)
            .agg(
                **{
                    "Late-delivery rate": ("is_late", "mean"),
                    "Eligible orders": ("order_id", "nunique"),
                }
            )
        )
        fig = px.line(
            monthly,
            x="Month",
            y="Late-delivery rate",
            markers=True,
            title="Late-delivery rate over time",
        )
        fig.update_traces(line_color=RED)
        fig.update_yaxes(tickformat=".0%")
        chart(styled(fig), "delivery-time")
    with right:
        state = rates(valid, "customer_state", "is_late", "Customer state").sort_values(
            "Rate"
        )
        fig = px.bar(
            state,
            x="Rate",
            y="Customer state",
            orientation="h",
            hover_data=["Eligible orders"],
            title="Late-delivery rate by customer state",
            color="Rate",
            color_continuous_scale="OrRd",
        )
        fig.update_xaxes(tickformat=".0%")
        fig.update_layout(coloraxis_showscale=False)
        chart(styled(fig, 500), "delivery-state")

    category = rates(
        valid, "product_category", "is_late", "Primary category", 100
    ).nlargest(15, "Rate")
    if category.empty:
        st.info(
            "No primary categories have at least 100 eligible delivered orders under these filters."
        )
    else:
        fig = px.bar(
            category.sort_values("Rate"),
            x="Rate",
            y="Primary category",
            orientation="h",
            hover_data=["Eligible orders"],
            title="Highest late-delivery rates by primary category (100+ eligible orders)",
            color_discrete_sequence=[RED],
        )
        fig.update_xaxes(tickformat=".0%")
        chart(styled(fig, 480), "delivery-category")


def lateness_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Lateness group"] = pd.cut(
        result["days_late"],
        bins=[-float("inf"), -1, 0, 2, 5, 10, float("inf")],
        labels=[
            "Early",
            "On time",
            "1–2 days late",
            "3–5 days late",
            "6–10 days late",
            "10+ days late",
        ],
        ordered=True,
    )
    return result


def customer_experience(frame: pd.DataFrame) -> None:
    st.header("Customer Experience")
    st.caption(
        "How strongly is customer feedback associated with delivery performance?"
    )
    reviewed = frame.dropna(subset=["review_score", "is_negative_review"]).copy()
    linked = lateness_groups(reviewed.dropna(subset=["days_late", "is_late"]))
    cols = st.columns(4)
    cols[0].metric("Reviewed orders", f"{reviewed['order_id'].nunique():,}")
    cols[1].metric("Average review score", number(reviewed["review_score"].mean(), 2))
    cols[2].metric(
        "Negative-review rate", percent(reviewed["is_negative_review"].mean())
    )
    cols[3].metric("Delivery-linked reviews", f"{linked['order_id'].nunique():,}")

    left, right = st.columns(2)
    with left:
        dist = (
            reviewed.groupby("review_score", as_index=False)["order_id"]
            .nunique()
            .rename(columns={"review_score": "Review score", "order_id": "Orders"})
        )
        if dist.empty:
            st.info("No reviews match the current filters.")
        else:
            fig = px.bar(
                dist,
                x="Review score",
                y="Orders",
                title="Review score distribution",
                color_discrete_sequence=[BLUE],
            )
            fig.update_xaxes(dtick=1)
            chart(styled(fig), "experience-reviews")
    with right:
        comparison = (
            linked.assign(
                **{
                    "Delivery outcome": linked["is_late"].map(
                        {False: "Not late", True: "Late"}
                    )
                }
            )
            .groupby("Delivery outcome", observed=True, as_index=False)
            .agg(
                **{
                    "Negative-review rate": ("is_negative_review", "mean"),
                    "Reviewed orders": ("order_id", "nunique"),
                }
            )
        )
        if comparison.empty:
            st.info(
                "No reviews with valid delivery outcomes match the current filters."
            )
        else:
            fig = px.bar(
                comparison,
                x="Delivery outcome",
                y="Negative-review rate",
                color="Delivery outcome",
                hover_data=["Reviewed orders"],
                title="Negative reviews: late vs not late deliveries",
                color_discrete_map={"Not late": GREEN, "Late": RED},
            )
            fig.update_yaxes(tickformat=".0%")
            chart(styled(fig), "experience-late")

    if linked.empty:
        st.info("No reviews with valid lateness outcomes match the current filters.")
    else:
        severity = (
            linked.dropna(subset=["Lateness group"])
            .groupby("Lateness group", observed=False, as_index=False)
            .agg(
                **{
                    "Negative-review rate": ("is_negative_review", "mean"),
                    "Average review score": ("review_score", "mean"),
                    "Reviewed orders": ("order_id", "nunique"),
                }
            )
        )
        fig = px.bar(
            severity,
            x="Lateness group",
            y="Negative-review rate",
            color="Negative-review rate",
            hover_data={"Average review score": ":.2f", "Reviewed orders": True},
            title="Negative-review rate rises with delivery delay",
            color_continuous_scale="OrRd",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(coloraxis_showscale=False)
        chart(styled(fig), "experience-severity")
        st.caption(
            "Negative review means a score of 1–2. Lateness uses whole-day differences: Early is below 0, "
            "On time is 0, and positive values are days after the estimate. Missing outcomes are excluded."
        )


def product_geography(frame: pd.DataFrame) -> None:
    st.header("Product & Geography")
    st.caption(
        "Where do order activity, value, delivery performance, and reviews differ?"
    )
    st.caption(
        "Product category is the primary (dominant) category per order. Category value charts group whole-order "
        "item value by that category; they are not exact item-level category revenue."
    )
    left, right = st.columns(2)
    with left:
        category = order_counts(frame, "product_category", "Primary category").nlargest(
            12, "Orders"
        )
        if category.empty:
            st.info("No primary categories match the current filters.")
        else:
            fig = px.bar(
                category.sort_values("Orders"),
                x="Orders",
                y="Primary category",
                orientation="h",
                title="Top primary categories by order count",
                color_discrete_sequence=[BLUE],
            )
            chart(styled(fig, 430), "product-orders")
    with right:
        values = (
            frame.dropna(subset=["product_category", "total_item_value"])
            .groupby("product_category", observed=True)["total_item_value"]
            .sum()
            .rename("Order item value")
            .nlargest(12)
            .reset_index()
            .rename(columns={"product_category": "Primary category"})
        )
        if values.empty:
            st.info("No category item values match the current filters.")
        else:
            fig = px.bar(
                values.sort_values("Order item value"),
                x="Order item value",
                y="Primary category",
                orientation="h",
                title="Order item value by primary category",
                color_discrete_sequence=[ORANGE],
            )
            fig.update_xaxes(tickprefix="R$", tickformat="~s")
            chart(styled(fig, 430), "product-values")

    left, right = st.columns(2)
    with left:
        state = order_counts(frame, "customer_state", "Customer state").sort_values(
            "Orders"
        )
        fig = px.bar(
            state,
            x="Orders",
            y="Customer state",
            orientation="h",
            title="Customer orders by state",
            color_discrete_sequence=[BLUE],
        )
        chart(styled(fig, 520), "product-customer-state")
    with right:
        seller = order_counts(
            frame, "seller_state", "Primary seller state"
        ).sort_values("Orders")
        if seller.empty:
            st.info("No primary seller states match the current filters.")
        else:
            fig = px.bar(
                seller,
                x="Orders",
                y="Primary seller state",
                orientation="h",
                title="Orders by primary seller state",
                color_discrete_sequence=[GREEN],
            )
            chart(styled(fig, 520), "product-seller-state")

    left, right = st.columns(2)
    with left:
        state_late = rates(
            frame, "customer_state", "is_late", "Customer state"
        ).sort_values("Rate")
        if state_late.empty:
            st.info("No valid state lateness outcomes match the current filters.")
        else:
            fig = px.bar(
                state_late,
                x="Rate",
                y="Customer state",
                orientation="h",
                hover_data=["Eligible orders"],
                title="Late-delivery rate by customer state",
                color_discrete_sequence=[RED],
            )
            fig.update_xaxes(tickformat=".0%")
            chart(styled(fig, 520), "product-state-late")
    with right:
        reviews = (
            frame.dropna(subset=["product_category", "review_score"])
            .groupby("product_category", observed=True)
            .agg(
                **{
                    "Average review score": ("review_score", "mean"),
                    "Reviewed orders": ("order_id", "nunique"),
                }
            )
            .reset_index()
            .rename(columns={"product_category": "Primary category"})
        )
        reviews = reviews[reviews["Reviewed orders"] >= 100].nsmallest(
            15, "Average review score"
        )
        if reviews.empty:
            st.info(
                "No primary categories have at least 100 reviewed orders under these filters."
            )
        else:
            fig = px.bar(
                reviews.sort_values("Average review score", ascending=False),
                x="Average review score",
                y="Primary category",
                orientation="h",
                hover_data=["Reviewed orders"],
                title="Lowest average reviews by primary category (100+ reviews)",
                color_discrete_sequence=[ORANGE],
            )
            fig.update_xaxes(range=[1, 5])
            chart(styled(fig, 520), "product-reviews")


def main() -> None:
    st.title("📦 Olist E-commerce Performance")
    st.caption(
        "Phase 1 descriptive dashboard · Historical Brazilian marketplace orders, 2016–2018"
    )
    try:
        data = load_data(str(DATA_PATH))
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Unable to load the dashboard dataset: {exc}")
        st.stop()

    st.sidebar.header("Dashboard")
    page = st.sidebar.radio("View", PAGES)
    st.sidebar.header("Global filters")
    minimum, maximum = (
        data["purchase_date"].min().date(),
        data["purchase_date"].max().date(),
    )
    date_range = st.sidebar.date_input(
        "Purchase date range",
        value=(minimum, maximum),
        min_value=minimum,
        max_value=maximum,
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
        data, date_range, customer_states, seller_states, categories, statuses
    )
    st.sidebar.caption(
        f"{filtered['order_id'].nunique():,} of {data['order_id'].nunique():,} orders selected"
    )
    if filtered.empty:
        st.warning(
            "No orders match the current filter combination. Adjust the filters to continue."
        )
        return

    {
        "Overview": overview,
        "Delivery Performance": delivery,
        "Customer Experience": customer_experience,
        "Product & Geography": product_geography,
    }[page](filtered)


if __name__ == "__main__":
    main()
