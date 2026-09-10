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
        "customer_city",
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
    frame["route_type"] = pd.NA
    has_route = frame["customer_state"].notna() & frame["seller_state"].notna()
    frame.loc[has_route, "route_type"] = (
        frame.loc[has_route, "customer_state"]
        .eq(frame.loc[has_route, "seller_state"])
        .map({True: "Same state", False: "Cross state"})
    )
    return frame


def options(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def apply_filters(
    frame: pd.DataFrame,
    date_range,
    customer_states,
    customer_cities,
    seller_states,
    route_types,
    categories,
    statuses,
) -> pd.DataFrame:
    """Apply dashboard filters with AND logic."""
    result = frame

    if len(date_range) == 2:
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(
            microseconds=1
        )
        result = result[result["purchase_date"].between(start, end)]

    for column, selected in (
        ("customer_state", customer_states),
        ("customer_city", customer_cities),
        ("seller_state", seller_states),
        ("route_type", route_types),
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


def add_bar_labels(
    figure: go.Figure,
    *,
    value_format: str = ",.0f",
    position: str = "auto",
) -> go.Figure:
    """Add readable direct labels to vertical bar traces."""
    figure.update_traces(
        texttemplate=f"%{{y:{value_format}}}",
        textposition=position,
        selector={"type": "bar"},
    )
    return figure


def delivered_orders(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["is_delivered_complete"]
        & frame["delivery_days"].notna()
        & frame["estimated_delivery_days"].notna()
        & frame["days_late"].notna()
        & frame["is_late"].notna()
    ].copy()


def page_insight(text: str) -> None:
    st.info(f"**What stands out:** {text}")


def pareto_chart(frame: pd.DataFrame) -> None:
    """Interactive category Pareto chart for business contribution."""
    st.subheader("Category contribution")
    metric = st.radio(
        "Pareto measure",
        ("Orders", "GMV"),
        horizontal=True,
        key="pareto-measure",
        help="Switch the Pareto view between order volume and gross merchandise value.",
    )

    base = frame.dropna(subset=["product_category"]).copy()
    if base.empty:
        st.info("No product categories match the current filters.")
        return

    if metric == "Orders":
        pareto = (
            base.groupby("product_category", observed=True)["order_id"]
            .nunique()
            .rename("Value")
            .sort_values(ascending=False)
            .reset_index()
        )
        axis_title = "Orders"
    else:
        pareto = (
            base.dropna(subset=["total_item_value"])
            .groupby("product_category", observed=True)["total_item_value"]
            .sum()
            .rename("Value")
            .sort_values(ascending=False)
            .reset_index()
        )
        axis_title = "GMV (R$)"

    pareto = pareto.head(15).copy()
    pareto["Cumulative share"] = pareto["Value"].cumsum() / pareto["Value"].sum()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pareto["product_category"],
            y=pareto["Value"],
            name=axis_title,
            marker_color=BLUE,
            text=pareto["Value"],
            texttemplate="%{text:,.0f}",
            textposition="outside",
            hovertemplate=(
                "%{x}<br>" + axis_title + ": %{y:,.0f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pareto["product_category"],
            y=pareto["Cumulative share"],
            name="Cumulative share",
            mode="lines+markers",
            yaxis="y2",
            line={"color": ORANGE, "width": 3},
            hovertemplate="%{x}<br>Cumulative share: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis={"title": axis_title, "tickformat": ","},
        yaxis2={
            "title": "Cumulative share",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".0%",
            "range": [0, 1.08],
        },
        xaxis={"title": "", "tickangle": -35},
        title=f"Top product categories by {metric.lower()}",
    )
    chart(styled(fig, 500), "overview-pareto")
    st.caption(
        "The Pareto view is descriptive business context. It shows concentration by category; "
        "it does not imply that category concentration causes delivery or review outcomes."
    )


def business_overview(frame: pd.DataFrame) -> None:
    st.header("Business Overview")
    st.caption(
        "Use the filters to compare how order volume, basket value, and category contribution change across segments."
    )

    reviewed = frame["review_score"].dropna()

    monthly = (
        frame.assign(Month=frame["purchase_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=False)
        .agg(
            Orders=("order_id", "nunique"),
            AOV=("total_item_value", "mean"),
        )
        .sort_values("Month")
    )

    if len(monthly) >= 2 and monthly.iloc[0]["Orders"] > 0 and monthly.iloc[0]["AOV"] > 0:
        order_change = monthly.iloc[-1]["Orders"] / monthly.iloc[0]["Orders"] - 1
        aov_change = monthly.iloc[-1]["AOV"] / monthly.iloc[0]["AOV"] - 1
        page_insight(
            f"From the first to last selected month, order volume changed by "
            f"{order_change:+.0%}, while average order value changed by {aov_change:+.0%}. "
            "This supports the EDA's volume-versus-basket comparison without assuming why volume changed."
        )
    else:
        page_insight(
            "The selected filters leave too little monthly history for a reliable first-versus-last comparison."
        )

    cols = st.columns(5)
    cols[0].metric("Orders", f"{frame['order_id'].nunique():,}")
    cols[1].metric("Unique customers", f"{frame['customer_unique_id'].nunique():,}")
    cols[2].metric(
        "Gross merchandise value",
        money(frame["total_item_value"].sum(min_count=1)),
        help="Sum of item prices, excluding freight. This is GMV, not Olist revenue.",
    )
    cols[3].metric("Average order value", money(frame["total_item_value"].mean()))
    cols[4].metric("Average review score", number(reviewed.mean(), 2))

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
        fig.update_yaxes(tickformat=",.0f")
        chart(styled(fig, 430), "overview-indexed-growth")

    left, right = st.columns(2)
    with left:
        orders_fig = px.bar(
            monthly,
            x="Month",
            y="Orders",
            title="Monthly order volume",
            color_discrete_sequence=[BLUE],
        )
        orders_fig.update_yaxes(tickformat=",")
        add_bar_labels(orders_fig)
        chart(styled(orders_fig), "overview-orders")

    with right:
        aov_fig = px.line(
            monthly,
            x="Month",
            y="AOV",
            markers=True,
            title="Monthly average order value",
            color_discrete_sequence=[ORANGE],
        )
        aov_fig.update_yaxes(tickprefix="R$", tickformat=",.0f")
        aov_fig.update_traces(
            mode="lines+markers+text",
            text=monthly["AOV"],
            texttemplate="R$%{text:,.0f}",
            textposition="top center",
        )
        chart(styled(aov_fig), "overview-aov")

    pareto_chart(frame)


def delivery_promise(frame: pd.DataFrame) -> None:
    st.header("Delivery Promise")
    st.caption(
        "Explore how the checkout promise compares with actual delivery across the selected customer, seller, route, and product segments."
    )

    valid = delivered_orders(frame)
    if valid.empty:
        st.info(
            "No delivered orders with valid quote and delivery timestamps match the filters."
        )
        return

    valid["days_early"] = -valid["days_late"].astype(float)
    quote_mae = valid["days_early"].abs().mean()
    median_actual = valid["delivery_days"].median()
    median_quote = valid["estimated_delivery_days"].median()
    median_early = valid["days_early"].median()

    if median_early >= 0:
        arrival_phrase = f"{median_early:,.0f} days earlier than promised"
    else:
        arrival_phrase = f"{abs(median_early):,.0f} days later than promised"

    page_insight(
        f"The median selected order was delivered in {median_actual:,.0f} days versus a "
        f"{median_quote:,.0f}-day quoted duration, arriving {arrival_phrase}. "
        "The comparison is descriptive of Olist's incumbent quote, not evidence of a replacement model."
    )

    cols = st.columns(4)
    cols[0].metric("Median actual delivery", f"{number(median_actual, 0)} days")
    cols[1].metric("Median quoted delivery", f"{number(median_quote, 0)} days")
    cols[2].metric(
        "Median arrival vs promise",
        f"{number(median_early, 0)} days early"
        if median_early >= 0
        else f"{number(abs(median_early), 0)} days late",
        help="Positive days-early values mean the order arrived before the estimated delivery date.",
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
            hover_data={"Orders": ":,"},
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
        bins = [-999, -30, -15, -7, -3, 0, 3, 7, 15, 999]
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
        valid["Delivery timing"] = pd.cut(
            valid["days_early"], bins=bins, labels=labels, ordered=True
        )
        timing_counts = (
            valid.dropna(subset=["Delivery timing"])
            .groupby("Delivery timing", observed=False)["order_id"]
            .nunique()
            .rename("Orders")
            .reset_index()
        )
        timing_counts["Delivery timing"] = timing_counts["Delivery timing"].astype(str)
        fig = px.bar(
            timing_counts,
            x="Delivery timing",
            y="Orders",
            title="Orders by delivery timing relative to the promise",
            color_discrete_sequence=[BLUE],
        )
        fig.update_yaxes(tickformat=",")
        add_bar_labels(fig)
        chart(styled(fig, 430), "promise-timing-counts")

    st.caption(
        f"Delivery metrics use {valid['order_id'].nunique():,} completed delivered orders "
        "with valid delivery and estimated-delivery timestamps under the current filters."
    )


def delivery_heatmap(valid: pd.DataFrame) -> None:
    st.subheader("Route heatmap")
    st.caption(
        "Cross-compare customer and seller states. Cells with too few eligible orders are omitted to reduce unstable rates."
    )

    metric = st.radio(
        "Heatmap measure",
        ("Late delivery rate", "Median delivery days"),
        horizontal=True,
        key="heatmap-measure",
    )
    minimum_orders = st.slider(
        "Minimum eligible orders per state pair",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        key="heatmap-minimum",
    )

    geography = valid.dropna(subset=["customer_state", "seller_state"]).copy()
    if geography.empty:
        st.info("No customer/seller geography is available under the current filters.")
        return

    grouped = (
        geography.groupby(["customer_state", "seller_state"], observed=True)
        .agg(
            eligible_orders=("order_id", "nunique"),
            late_rate=("is_late", "mean"),
            median_delivery_days=("delivery_days", "median"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["eligible_orders"] >= minimum_orders]

    if grouped.empty:
        st.info(
            "No customer/seller state pair meets the current minimum-order threshold."
        )
        return

    if metric == "Late delivery rate":
        value_col = "late_rate"
        fmt = ".1%"
        texttemplate = "%{z:.1%}"
        color_scale = "OrRd"
    else:
        value_col = "median_delivery_days"
        fmt = ".1f"
        texttemplate = "%{z:.1f}"
        color_scale = "Blues"

    pivot = grouped.pivot(
        index="customer_state",
        columns="seller_state",
        values=value_col,
    )
    counts = grouped.pivot(
        index="customer_state",
        columns="seller_state",
        values="eligible_orders",
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=color_scale,
            colorbar={"title": metric},
            customdata=counts.reindex(index=pivot.index, columns=pivot.columns).values,
            texttemplate=texttemplate,
            hovertemplate=(
                "Customer state: %{y}<br>"
                "Seller state: %{x}<br>"
                + metric
                + f": %{{z:{fmt}}}<br>"
                "Eligible orders: %{customdata:,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{metric} by customer state × seller state",
        xaxis_title="Seller state",
        yaxis_title="Customer state",
    )
    chart(styled(fig, 610), "experience-route-heatmap")


def delivery_experience(frame: pd.DataFrame) -> None:
    st.header("Delivery & Customer Experience")
    st.caption(
        "Compare delivery timing, review outcomes, and route geography across the same interactive filter context."
    )

    valid = delivered_orders(frame)
    reviewed = valid.dropna(subset=["review_score", "is_negative_review"]).copy()

    if reviewed.empty:
        st.info("No reviewed orders with valid delivery outcomes match the filters.")
        return

    reviewed["days_early"] = -reviewed["days_late"].astype(float)
    bins = [-999, -30, -15, -7, -3, 0, 3, 7, 15, 999]
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

    score_0_3 = timing.loc[
        timing["Delivery timing"].astype(str).eq("0–3 late"), "Average review score"
    ]
    score_3_7 = timing.loc[
        timing["Delivery timing"].astype(str).eq("3–7 late"), "Average review score"
    ]

    if not score_0_3.empty and not score_3_7.empty:
        page_insight(
            f"Average review score is {score_0_3.iloc[0]:.2f} for orders up to 3 days late "
            f"and {score_3_7.iloc[0]:.2f} for orders 3–7 days late under the current filters. "
            "The dashboard shows an association between delivery timing and reviews; it does not establish causality."
        )
    else:
        page_insight(
            "The selected filters leave too few observations in the key lateness bands for a useful comparison."
        )

    cols = st.columns(3)
    cols[0].metric("Reviewed delivered orders", f"{reviewed['order_id'].nunique():,}")
    cols[1].metric("Average review score", number(reviewed["review_score"].mean(), 2))
    cols[2].metric("Negative review rate", percent(reviewed["is_negative_review"].mean()))

    display_order = list(reversed(labels))
    timing["Delivery timing"] = pd.Categorical(
        timing["Delivery timing"].astype(str),
        categories=display_order,
        ordered=True,
    )
    timing = timing.sort_values("Delivery timing")

    fig = px.bar(
        timing,
        x="Delivery timing",
        y="Average review score",
        hover_data={
            "Negative review rate": ":.1%",
            "Reviewed orders": ":,",
        },
        title="Review score by delivery timing relative to the promise",
        color="Average review score",
        color_continuous_scale="RdYlGn",
        text="Reviewed orders",
    )
    fig.update_traces(
        texttemplate="%{y:.2f}<br>n=%{text:,.0f}",
        textposition="outside",
    )
    fig.update_yaxes(range=[1, 5.4])
    fig.update_layout(coloraxis_showscale=False)
    chart(styled(fig, 460), "experience-timing")

    geography = valid.dropna(subset=["route_type"]).copy()
    if not geography.empty:
        route = (
            geography.groupby("route_type", as_index=False)
            .agg(
                **{
                    "Median delivery days": ("delivery_days", "median"),
                    "Late delivery rate": ("is_late", "mean"),
                    "Eligible orders": ("order_id", "nunique"),
                }
            )
            .rename(columns={"route_type": "Route"})
        )

        left, right = st.columns(2)
        with left:
            fig = px.bar(
                route,
                x="Route",
                y="Median delivery days",
                hover_data={"Eligible orders": ":,"},
                title="Median delivery time by route type",
                color="Route",
                color_discrete_map={"Same state": GREEN, "Cross state": ORANGE},
            )
            add_bar_labels(fig, value_format=",.1f")
            chart(styled(fig), "experience-route-days")

        with right:
            fig = px.bar(
                route,
                x="Route",
                y="Late delivery rate",
                hover_data={"Eligible orders": ":,"},
                title="Late delivery rate by route type",
                color="Route",
                color_discrete_map={"Same state": GREEN, "Cross state": RED},
            )
            fig.update_traces(texttemplate="%{y:.1%}", textposition="outside")
            fig.update_yaxes(tickformat=".0%")
            chart(styled(fig), "experience-route-late")

    delivery_heatmap(valid)

    st.caption(
        "These views are descriptive associations, not causal estimates. Seller state is the "
        "dominant seller state for an order, so multi-seller orders are simplified for route comparisons."
    )


def render_filters(data: pd.DataFrame):
    minimum = data["purchase_date"].min().date()
    maximum = data["purchase_date"].max().date()
    default_start = max(ANALYSIS_START.date(), minimum)
    default_end = min(ANALYSIS_END.date(), maximum)

    st.sidebar.header("Explore the data")
    st.sidebar.caption(
        "Search within each multi-select. Leave a selector empty to include all values."
    )

    with st.sidebar.expander("Time", expanded=True):
        date_range = st.date_input(
            "Purchase date",
            value=(default_start, default_end),
            min_value=minimum,
            max_value=maximum,
            help="Defaults to the EDA analysis window: Jan 2017 to Aug 2018.",
        )

    with st.sidebar.expander("Customer geography", expanded=True):
        customer_states = st.multiselect(
            "State",
            options(data["customer_state"]),
            placeholder="All customer states",
        )
        city_base = data
        if customer_states:
            city_base = city_base[city_base["customer_state"].isin(customer_states)]
        customer_cities = st.multiselect(
            "City",
            options(city_base["customer_city"]),
            placeholder="All cities in selected state(s)",
            help="City choices are automatically narrowed by the selected customer state(s).",
        )

    with st.sidebar.expander("Route geography"):
        seller_states = st.multiselect(
            "Seller state",
            options(data["seller_state"]),
            placeholder="All seller states",
        )
        route_types = st.multiselect(
            "Route type",
            ["Same state", "Cross state"],
            placeholder="Both route types",
        )

    with st.sidebar.expander("Product"):
        categories = st.multiselect(
            "Product category",
            options(data["product_category"]),
            placeholder="All product categories",
        )

    with st.sidebar.expander("Order"):
        status_options = options(data["order_status"])
        status_labels = {
            status: status.replace("_", " ").title() for status in status_options
        }
        selected_status_labels = st.multiselect(
            "Order status",
            options=list(status_labels.values()),
            placeholder="All order statuses",
        )
        reverse_status = {label: status for status, label in status_labels.items()}
        statuses = [reverse_status[label] for label in selected_status_labels]

    return (
        date_range,
        customer_states,
        customer_cities,
        seller_states,
        route_types,
        categories,
        statuses,
    )


def main() -> None:
    st.title("📦 Olist E-commerce Analytics")
    st.caption(
        "Phase 1 interactive dashboard supporting the exploratory analysis and later problem scoping"
    )

    try:
        data = load_data(str(DATA_PATH))
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Unable to load the dashboard dataset: {exc}")
        st.stop()

    filter_values = render_filters(data)
    filtered = apply_filters(data, *filter_values)
    date_range = filter_values[0]

    if len(date_range) == 2:
        st.caption(
            f"**Analysis window:** {pd.Timestamp(date_range[0]).strftime('%b %Y')} to "
            f"{pd.Timestamp(date_range[1]).strftime('%b %Y')} · "
            f"**Selected:** {filtered['order_id'].nunique():,} of "
            f"{data['order_id'].nunique():,} orders"
        )

    if filtered.empty:
        st.warning("No orders match the current filter combination.")
        return

    overview_tab, promise_tab, experience_tab = st.tabs(
        [
            "Business Overview",
            "Delivery Promise",
            "Delivery & Customer Experience",
        ]
    )

    with overview_tab:
        business_overview(filtered)
    with promise_tab:
        delivery_promise(filtered)
    with experience_tab:
        delivery_experience(filtered)


if __name__ == "__main__":
    main()
