"""Phase 1 stakeholder dashboard aligned to the Olist exploratory analysis."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "dashboard_orders.parquet"
ANALYSIS_START = pd.Timestamp("2017-01-01")
ANALYSIS_END = pd.Timestamp("2018-08-31")

BLUE = "#2563EB"
ORANGE = "#F97316"
GREEN = "#0F766E"
RED = "#DC2626"
GREY = "#64748B"

BRAZIL_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast", "PB": "Northeast", "PE": "Northeast", "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}

PRODUCT_GROUPS = {
    "Home & Furniture": {
        "bed_bath_table", "furniture_decor", "housewares", "office_furniture", "home_construction",
        "home_confort", "home_comfort_2", "kitchen_dining_laundry_garden_furniture",
        "furniture_mattress_and_upholstery", "garden_tools", "construction_tools_construction",
        "construction_tools_lights", "construction_tools_safety", "air_conditioning", "la_cuisine",
    },
    "Electronics & Appliances": {
        "computers_accessories", "electronics", "telephony", "fixed_telephony", "computers",
        "tablets_printing_image", "home_appliances", "small_appliances", "small_appliances_home_oven_and_coffee",
        "audio", "consoles_games", "cine_photo",
    },
    "Fashion & Accessories": {
        "fashion_bags_accessories", "fashion_shoes", "fashion_underwear_beach", "fashion_male_clothing",
        "fashion_childrens_clothes", "fashion_sport", "watches_gifts", "luggage_accessories",
    },
    "Health & Beauty": {"health_beauty", "perfumery", "diapers_hygiene"},
    "Sports & Leisure": {"sports_leisure", "toys", "cool_stuff", "party_supplies", "christmas_supplies"},
    "Books, Media & Stationery": {
        "books_general_interest", "books_technical", "books_imported", "stationery", "music",
        "musical_instruments", "cds_dvds_musicals", "dvds_blu_ray", "art", "arts_and_craftmanship",
    },
    "Food & Beverage": {"food", "food_drink", "drinks"},
    "Baby & Kids": {"baby"},
    "Automotive": {"auto"},
    "Pets": {"pet_shop"},
    "Other": {"market_place", "signaling_security", "flowers"},
}

ORDER_STATUS_GROUPS = {
    "Fulfilled": {"delivered"},
    "In transit": {"shipped"},
    "In progress": {"created", "approved", "invoiced", "processing"},
    "Exception": {"canceled", "cancelled", "unavailable"},
}

st.set_page_config(page_title="Olist E-commerce Analytics", page_icon="📦", layout="wide")


def _lookup_group(value, mapping, default="Other"):
    if pd.isna(value):
        return pd.NA
    key = str(value)
    for group, members in mapping.items():
        if key in members:
            return group
    return default


def enrich_dimensions(frame: pd.DataFrame) -> pd.DataFrame:
    """Add presentation-only business groupings without changing the prepared dataset."""
    result = frame.copy()
    result["customer_region"] = result["customer_state"].map(BRAZIL_REGION)
    result["seller_region"] = result["seller_state"].map(BRAZIL_REGION)
    result["product_group"] = result["product_category"].map(lambda v: _lookup_group(v, PRODUCT_GROUPS))
    result["status_group"] = result["order_status"].map(lambda v: _lookup_group(v, ORDER_STATUS_GROUPS))
    result["route_type"] = pd.NA
    has_route = result["customer_state"].notna() & result["seller_state"].notna()
    result.loc[has_route, "route_type"] = (
        result.loc[has_route, "customer_state"].eq(result.loc[has_route, "seller_state"])
        .map({True: "Same state", False: "Cross state"})
    )
    return result


@st.cache_data(show_spinner="Loading dashboard data…")
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "order_id", "customer_unique_id", "purchase_date", "order_status", "customer_state",
        "customer_city", "seller_state", "product_category", "total_item_value", "review_score",
        "delivery_days", "estimated_delivery_days", "days_late", "is_delivered_complete",
        "is_late", "is_negative_review",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dashboard dataset is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["purchase_date"] = pd.to_datetime(frame["purchase_date"], errors="coerce")
    return enrich_dimensions(frame)


def options(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def apply_filters(
    frame: pd.DataFrame,
    date_range,
    customer_regions=None,
    customer_states=None,
    customer_cities=None,
    seller_regions=None,
    seller_states=None,
    route_types=None,
    product_groups=None,
    categories=None,
    status_groups=None,
    statuses=None,
) -> pd.DataFrame:
    """Apply all global selectors with AND logic, including parent and child groups."""
    result = enrich_dimensions(frame)
    selections = {
        "customer_region": customer_regions or [],
        "customer_state": customer_states or [],
        "customer_city": customer_cities or [],
        "seller_region": seller_regions or [],
        "seller_state": seller_states or [],
        "route_type": route_types or [],
        "product_group": product_groups or [],
        "product_category": categories or [],
        "status_group": status_groups or [],
        "order_status": statuses or [],
    }
    if len(date_range) == 2:
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        result = result[result["purchase_date"].between(start, end)]
    for column, selected in selections.items():
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
        margin={"l": 16, "r": 16, "t": 60, "b": 24},
        legend_title_text="",
        hoverlabel={"namelength": -1},
        font={"size": 13},
        uniformtext_minsize=11,
        uniformtext_mode="hide",
    )
    figure.update_xaxes(automargin=True)
    figure.update_yaxes(automargin=True)
    return figure


def chart(figure: go.Figure, key: str) -> None:
    st.plotly_chart(figure, use_container_width=True, key=key)


def add_bar_labels(
    figure: go.Figure,
    *,
    value_format: str = ",.0f",
    position: str = "auto",
    orientation: str = "v",
) -> go.Figure:
    value_axis = "x" if orientation == "h" else "y"
    figure.update_traces(
        texttemplate=f"%{{{value_axis}:{value_format}}}",
        textposition=position,
        cliponaxis=False,
        selector={"type": "bar"},
    )
    return figure


def delivered_orders(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["is_delivered_complete"] & frame["delivery_days"].notna()
        & frame["estimated_delivery_days"].notna() & frame["days_late"].notna() & frame["is_late"].notna()
    ].copy()


def page_insight(text: str) -> None:
    st.info(f"**What stands out:** {text}")


def pareto_chart(frame: pd.DataFrame) -> None:
    st.subheader("Category contribution")
    metric = st.radio(
        "Pareto measure", ("Orders", "GMV"), horizontal=True, key="pareto-measure",
        help="GMV means Gross Merchandise Value: the sum of item prices, excluding freight. It is not Olist revenue.",
    )
    base = frame.dropna(subset=["product_category"]).copy()
    if base.empty:
        st.info("No product categories match the current filters.")
        return
    if metric == "Orders":
        pareto = base.groupby("product_category")["order_id"].nunique().rename("Value").sort_values(ascending=False).reset_index()
        axis_title, text_template, hover_template = "Orders", "%{text:,.0f}", "%{x}<br>Orders: %{y:,.0f}<extra></extra>"
    else:
        pareto = base.dropna(subset=["total_item_value"]).groupby("product_category")["total_item_value"].sum().rename("Value").sort_values(ascending=False).reset_index()
        axis_title, text_template, hover_template = "Gross merchandise value (R$)", "R$%{text:,.0f}", "%{x}<br>Gross merchandise value: R$%{y:,.0f}<extra></extra>"
    total_value = pareto["Value"].sum()
    display = pareto.head(10).copy()
    others_value = pareto.iloc[10:]["Value"].sum()
    if others_value > 0:
        display = pd.concat([display, pd.DataFrame({"product_category": ["Others"], "Value": [others_value]})], ignore_index=True)
    display["Cumulative share"] = display["Value"].cumsum() / total_value
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=display["product_category"], y=display["Value"], name=axis_title, marker_color=BLUE,
        text=display["Value"], texttemplate=text_template, textposition="outside", cliponaxis=False,
        hovertemplate=hover_template,
    ))
    fig.add_trace(go.Scatter(
        x=display["product_category"], y=display["Cumulative share"], name="Cumulative share",
        mode="lines+markers", yaxis="y2", line={"color": ORANGE, "width": 3},
        hovertemplate="%{x}<br>Cumulative share: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        yaxis={"title": axis_title, "tickformat": ","},
        yaxis2={"title": "Cumulative share", "overlaying": "y", "side": "right", "tickformat": ".0%", "range": [0, 1.08]},
        xaxis={"title": "", "tickangle": -35}, title=f"Top 10 product categories + Others by {metric.lower()}",
    )
    chart(styled(fig, 500), "overview-pareto")
    st.caption("Top 10 categories stay visible; the remaining long tail is combined into Others. The cumulative line uses the full selected population.")


def business_overview(frame: pd.DataFrame) -> None:
    st.header("Business Overview")
    st.caption("Use the filters to compare order volume, basket value, and category contribution across segments.")
    monthly = (
        frame.assign(Month=frame["purchase_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=False).agg(Orders=("order_id", "nunique"), AOV=("total_item_value", "mean")).sort_values("Month")
    )
    if len(monthly) >= 2 and monthly.iloc[0]["Orders"] > 0 and monthly.iloc[0]["AOV"] > 0:
        page_insight(
            f"From the first to last selected month, order volume changed by {monthly.iloc[-1]['Orders'] / monthly.iloc[0]['Orders'] - 1:+.0%}, "
            f"while average order value changed by {monthly.iloc[-1]['AOV'] / monthly.iloc[0]['AOV'] - 1:+.0%}."
        )
    else:
        page_insight("The selected filters leave too little monthly history for a reliable first-versus-last comparison.")

    reviewed = frame["review_score"].dropna()
    cols = st.columns(5)
    cols[0].metric("Orders", f"{frame['order_id'].nunique():,}")
    cols[1].metric("Unique customers", f"{frame['customer_unique_id'].nunique():,}")
    cols[2].metric(
        "GMV", money(frame["total_item_value"].sum(min_count=1)),
        help="GMV = Gross Merchandise Value. Here it is the sum of item prices, excluding freight; it is not Olist revenue.",
    )
    cols[3].metric(
        "AOV", money(frame["total_item_value"].mean()),
        help="AOV = Average Order Value. Here it is the mean item value per order in the prepared one-row-per-order dataset.",
    )
    cols[4].metric("Average review score", number(reviewed.mean(), 2))
    if monthly.empty:
        return

    left, right = st.columns(2)
    with left:
        fig = px.bar(monthly, x="Month", y="Orders", title="Monthly order volume", color_discrete_sequence=[BLUE])
        fig.update_yaxes(title="Orders", tickformat=",")
        fig.update_traces(
            text=monthly["Orders"], texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False,
            hovertemplate="%{x|%b %Y}<br>Orders: %{y:,.0f}<extra></extra>",
        )
        chart(styled(fig, 420), "overview-orders")
    with right:
        fig = px.line(monthly, x="Month", y="AOV", markers=True, title="Monthly average order value", color_discrete_sequence=[ORANGE])
        fig.update_yaxes(title="Average order value (R$)", tickprefix="R$", tickformat=",.0f")
        fig.update_traces(
            mode="lines+markers",
            hovertemplate="%{x|%b %Y}<br>Average order value: R$%{y:,.2f}<extra></extra>",
        )
        latest = monthly.dropna(subset=["AOV"]).tail(1)
        if not latest.empty:
            fig.add_trace(go.Scatter(
                x=latest["Month"], y=latest["AOV"], mode="text", showlegend=False,
                text=[f"R${latest['AOV'].iloc[0]:,.0f}"], textposition="top center",
                hoverinfo="skip",
            ))
        chart(styled(fig, 420), "overview-aov")
    pareto_chart(frame)


def delivery_promise(frame: pd.DataFrame) -> None:
    st.header("Delivery Promise")
    st.caption("Compare Olist's checkout promise with actual delivery across the selected segment.")
    valid = delivered_orders(frame)
    if valid.empty:
        st.info("No delivered orders with valid quote and delivery timestamps match the filters.")
        return

    valid["days_early"] = -valid["days_late"].astype(float)
    median_actual = valid["delivery_days"].median()
    median_quote = valid["estimated_delivery_days"].median()
    median_early = valid["days_early"].median()
    quote_mae = valid["days_early"].abs().mean()
    page_insight(
        f"Median actual delivery is {median_actual:,.0f} days versus {median_quote:,.0f} quoted days; "
        f"the median order arrived {abs(median_early):,.0f} days {'early' if median_early >= 0 else 'late'}."
    )

    cols = st.columns(4)
    cols[0].metric("Median actual delivery", f"{number(median_actual, 0)} days")
    cols[1].metric("Median quoted delivery", f"{number(median_quote, 0)} days")
    cols[2].metric(
        "Median vs promise", f"{number(abs(median_early), 0)}d {'early' if median_early >= 0 else 'late'}",
        help="Difference between the actual delivery date and the checkout promise, summarised by the median order.",
    )
    cols[3].metric(
        "Quote MAE", f"{number(quote_mae, 1)} days",
        help="MAE = Mean Absolute Error. It is the average absolute gap between actual and promised delivery timing; lower means the promise is closer to reality.",
    )

    left, right = st.columns(2)
    with left:
        quote = (
            valid.groupby("estimated_delivery_days", as_index=False)
            .agg(**{"Median actual delivery days": ("delivery_days", "median"), "Orders": ("order_id", "nunique")})
            .sort_values("estimated_delivery_days")
        )
        axis_max = max(float(quote["estimated_delivery_days"].max()), float(quote["Median actual delivery days"].max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=quote["estimated_delivery_days"], y=quote["Median actual delivery days"],
            mode="lines+markers", name="Median actual delivery", customdata=quote[["Orders"]],
            line={"color": ORANGE, "width": 3},
            hovertemplate="Quoted: %{x:.0f} days<br>Median actual: %{y:.1f} days<br>Orders: %{customdata[0]:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[0, axis_max], y=[0, axis_max], mode="lines", name="Actual = promised",
            line={"color": GREY, "dash": "dash"}, hoverinfo="skip",
        ))
        fig.update_layout(
            title="Are delivery promises conservative or optimistic?",
            xaxis_title="Promised delivery days",
            yaxis_title="Median actual delivery days",
        )
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
            text="Below dashed line = earlier than promised<br>Above dashed line = later than promised",
            bgcolor="rgba(255,255,255,0.85)", bordercolor=GREY, font={"size": 11},
        )
        chart(styled(fig, 450), "promise-quoted-actual")

    with right:
        bins = [-999, -30, -15, -7, -3, 0, 3, 7, 15, 999]
        labels = ["30+ late", "15–30 late", "7–15 late", "3–7 late", "0–3 late", "0–3 early", "3–7 early", "7–15 early", "15+ early"]
        valid["Delivery timing"] = pd.cut(valid["days_early"], bins=bins, labels=labels, ordered=True)
        counts = (
            valid.dropna(subset=["Delivery timing"])
            .groupby("Delivery timing", observed=False)["order_id"].nunique().rename("Orders").reset_index()
        )
        counts["Delivery timing"] = counts["Delivery timing"].astype(str)
        counts["Side"] = counts["Delivery timing"].str.contains("late").map({True: "Late", False: "Early"})
        counts["Signed orders"] = counts["Orders"].where(counts["Side"].eq("Early"), -counts["Orders"])
        counts["Band"] = counts["Delivery timing"].str.replace(" late", "", regex=False).str.replace(" early", "", regex=False)
        display_order = ["30+", "15–30", "7–15", "3–7", "0–3"]
        fig = go.Figure()
        late = counts[counts["Side"] == "Late"].copy()
        early = counts[counts["Side"] == "Early"].copy()
        fig.add_trace(go.Bar(
            y=late["Band"], x=late["Signed orders"], orientation="h", name="Late", marker_color=RED,
            customdata=late[["Orders", "Delivery timing"]], text=late["Orders"],
            texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False,
            hovertemplate="%{customdata[1]}<br>Orders: %{customdata[0]:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=early["Band"], x=early["Signed orders"], orientation="h", name="Early", marker_color=GREEN,
            customdata=early[["Orders", "Delivery timing"]], text=early["Orders"],
            texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False,
            hovertemplate="%{customdata[1]}<br>Orders: %{customdata[0]:,.0f}<extra></extra>",
        ))
        max_orders = max(float(counts["Orders"].max()), 1.0)
        fig.update_layout(
            title="Early vs late delivery relative to the promise",
            barmode="relative",
            xaxis={
                "title": "Orders (late ← promise → early)",
                "range": [-max_orders * 1.25, max_orders * 1.25],
                "tickformat": ",",
            },
            yaxis={"title": "Days from promise", "categoryorder": "array", "categoryarray": display_order[::-1]},
        )
        fig.add_vline(x=0, line_width=2, line_color=GREY)
        chart(styled(fig, 450), "promise-timing-counts")

    st.caption(f"Delivery metrics use {valid['order_id'].nunique():,} eligible completed delivered orders under the current filters.")


def delivery_heatmap(valid: pd.DataFrame) -> None:
    st.subheader("Route heatmap")
    metric = st.radio("Heatmap measure", ("Late delivery rate", "Median delivery days"), horizontal=True, key="heatmap-measure")
    minimum_orders = st.slider("Minimum eligible orders per state pair", 10, 200, 50, 10, key="heatmap-minimum")
    geography = valid.dropna(subset=["customer_state", "seller_state"]).copy()
    grouped = geography.groupby(["customer_state", "seller_state"]).agg(
        eligible_orders=("order_id", "nunique"), late_rate=("is_late", "mean"), median_delivery_days=("delivery_days", "median")
    ).reset_index()
    grouped = grouped[grouped["eligible_orders"] >= minimum_orders]
    if grouped.empty:
        st.info("No customer/seller state pair meets the current minimum-order threshold.")
        return
    value_col, fmt, texttemplate, color_scale = (
        ("late_rate", ".1%", "%{z:.1%}", "OrRd") if metric == "Late delivery rate"
        else ("median_delivery_days", ".1f", "%{z:.1f}", "Blues")
    )
    pivot = grouped.pivot(index="customer_state", columns="seller_state", values=value_col)
    counts = grouped.pivot(index="customer_state", columns="seller_state", values="eligible_orders")
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(), colorscale=color_scale,
        customdata=counts.reindex(index=pivot.index, columns=pivot.columns).values, texttemplate=texttemplate,
        hovertemplate="Customer state: %{y}<br>Seller state: %{x}<br>" + metric + f": %{{z:{fmt}}}<br>Eligible orders: %{{customdata:,.0f}}<extra></extra>",
    ))
    fig.update_layout(title=f"{metric} by customer state × seller state", xaxis_title="Seller state", yaxis_title="Customer state")
    chart(styled(fig, 610), "experience-route-heatmap")


def delivery_experience(frame: pd.DataFrame) -> None:
    st.header("Delivery & Customer Experience")
    st.caption("Compare delivery timing, review outcomes, and route geography across the same filter context.")
    valid = delivered_orders(frame)
    reviewed = valid.dropna(subset=["review_score", "is_negative_review"]).copy()
    if reviewed.empty:
        st.info("No reviewed orders with valid delivery outcomes match the filters.")
        return
    reviewed["days_early"] = -reviewed["days_late"].astype(float)
    bins = [-999, -30, -15, -7, -3, 0, 3, 7, 15, 999]
    labels = ["30+ late", "15–30 late", "7–15 late", "3–7 late", "0–3 late", "0–3 early", "3–7 early", "7–15 early", "15+ early"]
    reviewed["Delivery timing"] = pd.cut(reviewed["days_early"], bins=bins, labels=labels, ordered=True)
    timing = reviewed.dropna(subset=["Delivery timing"]).groupby("Delivery timing", observed=False, as_index=False).agg(
        **{"Average review score": ("review_score", "mean"), "Negative review rate": ("is_negative_review", "mean"), "Reviewed orders": ("order_id", "nunique")}
    )
    page_insight("Review outcomes deteriorate as orders move further past the promised delivery date; use the timing bands below to compare the selected segment.")
    cols = st.columns(3)
    cols[0].metric("Reviewed delivered orders", f"{reviewed['order_id'].nunique():,}")
    cols[1].metric("Average review score", number(reviewed["review_score"].mean(), 2))
    cols[2].metric("Negative review rate", percent(reviewed["is_negative_review"].mean()))
    timing["Delivery timing"] = pd.Categorical(timing["Delivery timing"].astype(str), categories=list(reversed(labels)), ordered=True)
    timing = timing.sort_values("Delivery timing")
    fig = px.bar(
        timing, x="Delivery timing", y="Average review score",
        hover_data={"Negative review rate": ":.1%", "Reviewed orders": ":,"},
        title="Review score by delivery timing relative to the promise",
        color="Average review score", color_continuous_scale="RdYlGn", text="Reviewed orders",
    )
    fig.update_traces(texttemplate="%{y:.2f}<br>n=%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(range=[1, 5.4])
    fig.update_layout(coloraxis_showscale=False)
    chart(styled(fig, 460), "experience-timing")

    geography = valid.dropna(subset=["route_type"]).copy()
    if not geography.empty:
        route = geography.groupby("route_type", as_index=False).agg(
            **{"Median delivery days": ("delivery_days", "median"), "Late delivery rate": ("is_late", "mean"), "Eligible orders": ("order_id", "nunique")}
        ).rename(columns={"route_type": "Route"})
        left, right = st.columns(2)
        with left:
            fig = px.bar(
                route, x="Route", y="Median delivery days", hover_data={"Eligible orders": ":,"},
                title="Median delivery time by route type", color="Route",
                color_discrete_map={"Same state": GREEN, "Cross state": ORANGE},
            )
            add_bar_labels(fig, value_format=",.1f")
            chart(styled(fig), "experience-route-days")
        with right:
            fig = px.bar(
                route, x="Route", y="Late delivery rate", hover_data={"Eligible orders": ":,"},
                title="Late delivery rate by route type", color="Route",
                color_discrete_map={"Same state": GREEN, "Cross state": RED},
            )
            fig.update_traces(texttemplate="%{y:.1%}", textposition="outside", cliponaxis=False)
            fig.update_yaxes(tickformat=".0%")
            chart(styled(fig), "experience-route-late")
    delivery_heatmap(valid)
    st.caption("These are descriptive associations, not causal estimates. Seller state is the dominant seller state for an order.")


def _child_options(data: pd.DataFrame, parent_column: str, parents, child_column: str) -> list[str]:
    subset = data[data[parent_column].isin(parents)] if parents else data
    return options(subset[child_column])


def render_filters(data: pd.DataFrame):
    data = enrich_dimensions(data)
    minimum, maximum = data["purchase_date"].min().date(), data["purchase_date"].max().date()
    default_start, default_end = max(ANALYSIS_START.date(), minimum), min(ANALYSIS_END.date(), maximum)
    st.sidebar.header("Filters & Segments")
    st.sidebar.caption("Choose a high-level segment, then refine it with the child selectors. Leave any selector empty to include all values.")
    with st.sidebar.expander("Time", expanded=True):
        date_range = st.date_input("Purchase date", value=(default_start, default_end), min_value=minimum, max_value=maximum)
    with st.sidebar.expander("Customer location", expanded=True):
        customer_regions = st.multiselect("Region", options(data["customer_region"]), placeholder="All customer regions")
        customer_states = st.multiselect("State", _child_options(data, "customer_region", customer_regions, "customer_state"), placeholder="All states in selected region(s)")
        state_base = data[data["customer_region"].isin(customer_regions)] if customer_regions else data
        customer_cities = st.multiselect("City", _child_options(state_base, "customer_state", customer_states, "customer_city"), placeholder="All cities in selected state(s)")
    with st.sidebar.expander("Delivery route"):
        route_types = st.multiselect("Route type", ["Same state", "Cross state"], placeholder="Both route types")
        seller_regions = st.multiselect("Seller region", options(data["seller_region"]), placeholder="All seller regions")
        seller_states = st.multiselect("Seller state", _child_options(data, "seller_region", seller_regions, "seller_state"), placeholder="All states in selected seller region(s)")
    with st.sidebar.expander("Product"):
        product_groups = st.multiselect("Product group", options(data["product_group"]), placeholder="All product groups")
        categories = st.multiselect("Product category", _child_options(data, "product_group", product_groups, "product_category"), placeholder="All categories in selected group(s)")
    with st.sidebar.expander("Order lifecycle"):
        status_groups = st.multiselect("Status group", options(data["status_group"]), placeholder="All status groups")
        status_options = _child_options(data, "status_group", status_groups, "order_status")
        status_labels = {status: status.replace("_", " ").title() for status in status_options}
        selected_labels = st.multiselect("Order status", list(status_labels.values()), placeholder="All statuses in selected group(s)")
        reverse = {label: status for status, label in status_labels.items()}
        statuses = [reverse[label] for label in selected_labels]
    return (
        date_range, customer_regions, customer_states, customer_cities, seller_regions, seller_states,
        route_types, product_groups, categories, status_groups, statuses,
    )


def main() -> None:
    st.title("📦 Olist E-commerce Analytics")
    st.caption("Phase 1 interactive dashboard supporting exploratory analysis and later problem scoping")
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
            f"**Analysis window:** {pd.Timestamp(date_range[0]).strftime('%b %Y')} to {pd.Timestamp(date_range[1]).strftime('%b %Y')} · "
            f"**Selected:** {filtered['order_id'].nunique():,} of {data['order_id'].nunique():,} orders"
        )
    if filtered.empty:
        st.warning("No orders match the current filter combination.")
        return
    overview_tab, promise_tab, experience_tab = st.tabs(["Business Overview", "Delivery Promise", "Delivery & Customer Experience"])
    with overview_tab:
        business_overview(filtered)
    with promise_tab:
        delivery_promise(filtered)
    with experience_tab:
        delivery_experience(filtered)


if __name__ == "__main__":
    main()
