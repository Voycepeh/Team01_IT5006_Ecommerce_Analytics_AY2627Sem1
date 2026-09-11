"""Phase 1 stakeholder dashboard aligned to the Olist exploratory analysis."""

from pathlib import Path

import pandas as pd
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
YEAR_COLORS = [BLUE, ORANGE, GREEN, GREY, RED]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BRAZIL_REGION = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast", "PB": "Northeast", "PE": "Northeast", "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Central-West", "GO": "Central-West", "MT": "Central-West", "MS": "Central-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}
REGION_ORDER = ["North", "Northeast", "Central-West", "Southeast", "South"]

PRODUCT_GROUPS = {
    "Home & Furniture": {"bed_bath_table", "furniture_decor", "housewares", "office_furniture", "home_construction", "home_confort", "home_comfort_2", "kitchen_dining_laundry_garden_furniture", "furniture_mattress_and_upholstery", "garden_tools", "construction_tools_construction", "construction_tools_lights", "construction_tools_safety", "air_conditioning", "la_cuisine"},
    "Electronics & Appliances": {"computers_accessories", "electronics", "telephony", "fixed_telephony", "computers", "tablets_printing_image", "home_appliances", "small_appliances", "small_appliances_home_oven_and_coffee", "audio", "consoles_games", "cine_photo"},
    "Fashion & Accessories": {"fashion_bags_accessories", "fashion_shoes", "fashion_underwear_beach", "fashion_male_clothing", "fashion_childrens_clothes", "fashion_sport", "watches_gifts", "luggage_accessories"},
    "Health & Beauty": {"health_beauty", "perfumery", "diapers_hygiene"},
    "Sports & Leisure": {"sports_leisure", "toys", "cool_stuff", "party_supplies", "christmas_supplies"},
    "Books, Media & Stationery": {"books_general_interest", "books_technical", "books_imported", "stationery", "music", "musical_instruments", "cds_dvds_musicals", "dvds_blu_ray", "art", "arts_and_craftmanship"},
    "Food & Beverage": {"food", "food_drink", "drinks"},
    "Baby & Kids": {"baby"}, "Automotive": {"auto"}, "Pets": {"pet_shop"},
    "Other": {"market_place", "signaling_security", "flowers"},
}
ORDER_STATUS_GROUPS = {
    "Fulfilled": {"delivered"}, "In transit": {"shipped"},
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
    result = frame.copy()
    result["customer_region"] = result["customer_state"].map(BRAZIL_REGION)
    result["seller_region"] = result["seller_state"].map(BRAZIL_REGION)
    result["product_group"] = result["product_category"].map(lambda v: _lookup_group(v, PRODUCT_GROUPS))
    result["status_group"] = result["order_status"].map(lambda v: _lookup_group(v, ORDER_STATUS_GROUPS))
    result["route_type"] = pd.NA
    has_route = result["customer_state"].notna() & result["seller_state"].notna()
    result.loc[has_route, "route_type"] = result.loc[has_route, "customer_state"].eq(result.loc[has_route, "seller_state"]).map({True: "Same state", False: "Cross state"})
    return result


@st.cache_data(show_spinner="Loading dashboard data…")
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"order_id", "customer_unique_id", "purchase_date", "order_status", "customer_state", "customer_city", "seller_state", "product_category", "total_item_value", "review_score", "delivery_days", "estimated_delivery_days", "days_late", "is_delivered_complete", "is_late", "is_negative_review"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dashboard dataset is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["purchase_date"] = pd.to_datetime(frame["purchase_date"], errors="coerce")
    return enrich_dimensions(frame)


def options(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def apply_filters(frame: pd.DataFrame, date_range, customer_regions=None, customer_states=None, customer_cities=None, seller_regions=None, seller_states=None, route_types=None, product_groups=None, categories=None, status_groups=None, statuses=None) -> pd.DataFrame:
    result = enrich_dimensions(frame)
    selections = {
        "customer_region": customer_regions or [], "customer_state": customer_states or [], "customer_city": customer_cities or [],
        "seller_region": seller_regions or [], "seller_state": seller_states or [], "route_type": route_types or [],
        "product_group": product_groups or [], "product_category": categories or [], "status_group": status_groups or [], "order_status": statuses or [],
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
    if pd.isna(value): return "—"
    if abs(value) >= 1_000_000: return f"R${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000: return f"R${value / 1_000:,.1f}K"
    return f"R${value:,.2f}"


def number(value: float, decimals: int = 1) -> str:
    return "—" if pd.isna(value) else f"{value:,.{decimals}f}"


def percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def styled(figure: go.Figure, height: int = 390) -> go.Figure:
    figure.update_layout(template="plotly_white", height=height, margin={"l": 16, "r": 16, "t": 60, "b": 24}, legend_title_text="", hoverlabel={"namelength": -1}, font={"size": 13}, uniformtext_minsize=11, uniformtext_mode="hide")
    figure.update_xaxes(automargin=True)
    figure.update_yaxes(automargin=True)
    return figure


def chart(figure: go.Figure, key: str) -> None:
    st.plotly_chart(figure, use_container_width=True, key=key)


def delivered_orders(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["is_delivered_complete"] & frame["delivery_days"].notna() & frame["estimated_delivery_days"].notna() & frame["days_late"].notna() & frame["is_late"].notna()].copy()


def page_insight(text: str) -> None:
    st.info(f"**What stands out:** {text}")


def monthly_business_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(Month=frame["purchase_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Month", as_index=False)
        .agg(Orders=("order_id", "nunique"), GMV=("total_item_value", "sum"), AOV=("total_item_value", "mean"))
        .sort_values("Month")
    )


def yoy_metric_figure(monthly: pd.DataFrame, metric: str, title: str) -> go.Figure:
    data = monthly.copy()
    data["Year"] = data["Month"].dt.year
    data["Month number"] = data["Month"].dt.month
    data["Month label"] = data["Month"].dt.strftime("%b")
    fig = go.Figure()
    for index, year in enumerate(sorted(data["Year"].unique())):
        year_data = data[data["Year"] == year].sort_values("Month number")
        if metric == "Orders":
            hover = f"%{{x}}<br>Orders: %{{y:,.0f}}<extra>{year}</extra>"
        else:
            hover = f"%{{x}}<br>{metric}: R$%{{y:,.2f}}<extra>{year}</extra>"
        fig.add_trace(go.Scatter(
            x=year_data["Month label"], y=year_data[metric], mode="lines+markers", name=str(year),
            line={"width": 2.5, "color": YEAR_COLORS[index % len(YEAR_COLORS)]},
            marker={"size": 6}, hovertemplate=hover,
        ))
    y_title = "Orders" if metric == "Orders" else ("Gross merchandise value (R$)" if metric == "GMV" else "Average order value (R$)")
    fig.update_layout(
        title=title,
        xaxis={"title": "Month", "categoryorder": "array", "categoryarray": MONTH_ORDER},
        yaxis={"title": y_title, "tickformat": ","},
        legend={"title": "Year", "orientation": "h", "y": 1.16, "x": 0.5, "xanchor": "center"},
    )
    if metric in {"GMV", "AOV"}:
        fig.update_yaxes(tickprefix="R$")
    return styled(fig, 330)


def contribution_figure(frame: pd.DataFrame, dimension: str, metric: str, title: str) -> go.Figure:
    base = frame.dropna(subset=[dimension]).copy()
    if metric == "Orders":
        grouped = base.groupby(dimension)["order_id"].nunique().rename("Value").sort_values(ascending=False).reset_index()
        axis_title, texttemplate, hover = "Orders", "%{text:,.0f}", "%{x}<br>Orders: %{y:,.0f}<extra></extra>"
    elif metric == "GMV":
        grouped = base.dropna(subset=["total_item_value"]).groupby(dimension)["total_item_value"].sum().rename("Value").sort_values(ascending=False).reset_index()
        axis_title, texttemplate, hover = "Gross merchandise value (R$)", "R$%{text:,.0f}", "%{x}<br>GMV: R$%{y:,.0f}<extra></extra>"
    else:
        grouped = base.dropna(subset=["total_item_value"]).groupby(dimension)["total_item_value"].mean().rename("Value").sort_values(ascending=False).reset_index()
        axis_title, texttemplate, hover = "Average order value (R$)", "R$%{text:,.0f}", "%{x}<br>AOV: R$%{y:,.2f}<extra></extra>"
    if grouped.empty:
        return go.Figure()
    if metric in {"Orders", "GMV"}:
        display = grouped.head(10).copy()
        others = grouped.iloc[10:]["Value"].sum()
        if others > 0:
            display = pd.concat([display, pd.DataFrame({dimension: ["Others"], "Value": [others]})], ignore_index=True)
        display["Cumulative share"] = display["Value"].cumsum() / grouped["Value"].sum()
    else:
        display = grouped.head(10).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=display[dimension], y=display["Value"], name=axis_title, marker_color=BLUE,
        text=display["Value"], texttemplate=texttemplate, textposition="outside", cliponaxis=False, hovertemplate=hover,
    ))
    if metric in {"Orders", "GMV"}:
        fig.add_trace(go.Scatter(
            x=display[dimension], y=display["Cumulative share"], yaxis="y2", mode="lines+markers",
            name="Cumulative share", line={"color": ORANGE, "width": 3},
            hovertemplate="%{x}<br>Cumulative share: %{y:.1%}<extra></extra>",
        ))
        fig.update_layout(yaxis2={"title": "Cumulative share", "overlaying": "y", "side": "right", "tickformat": ".0%", "range": [0, 1.08]})
    fig.update_layout(
        title=title,
        xaxis={"title": "", "tickangle": -30},
        yaxis={"title": axis_title, "rangemode": "tozero"},
        legend={"orientation": "h", "y": 1.13, "x": 0.5, "xanchor": "center"},
    )
    if metric in {"GMV", "AOV"}:
        fig.update_yaxes(tickprefix="R$")
    return styled(fig, 520)


def _centered_pills(label: str, choices: tuple[str, ...], key: str, default: str) -> str:
    columns = st.columns(3)
    with columns[1]:
        if hasattr(st, "segmented_control"):
            selected = st.segmented_control(label, choices, default=default, key=key, label_visibility="collapsed")
            return selected or default
        return st.radio(label, choices, horizontal=True, key=key, label_visibility="collapsed")


def business_overview(frame: pd.DataFrame) -> None:
    st.header("Overview")
    st.caption("Track Orders, GMV and AOV over time, then see where business activity is concentrated by customer geography and product.")
    monthly = monthly_business_metrics(frame)
    if monthly.empty:
        st.info("No monthly business activity matches the current filters.")
        return
    if len(monthly) >= 2 and monthly.iloc[0]["Orders"] > 0 and monthly.iloc[0]["GMV"] > 0 and monthly.iloc[0]["AOV"] > 0:
        page_insight(
            f"Across the selected period, Orders changed by {monthly.iloc[-1]['Orders'] / monthly.iloc[0]['Orders'] - 1:+.0%}, "
            f"GMV by {monthly.iloc[-1]['GMV'] / monthly.iloc[0]['GMV'] - 1:+.0%}, and AOV by {monthly.iloc[-1]['AOV'] / monthly.iloc[0]['AOV'] - 1:+.0%} from the first to last selected month."
        )

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Orders", f"{frame['order_id'].nunique():,}", border=True)
    kpi_cols[1].metric("GMV", money(frame["total_item_value"].sum(min_count=1)), help="GMV = Gross Merchandise Value. Here it is the sum of item prices, excluding freight; it is not Olist revenue.", border=True)
    kpi_cols[2].metric("AOV", money(frame["total_item_value"].mean()), help="AOV = Average Order Value. Here it is the mean item value per order in the prepared one-row-per-order dataset.", border=True)

    trend_cols = st.columns(3)
    with trend_cols[0]:
        chart(yoy_metric_figure(monthly, "Orders", "Monthly Orders by year"), "overview-orders-yoy")
    with trend_cols[1]:
        chart(yoy_metric_figure(monthly, "GMV", "Monthly GMV by year"), "overview-gmv-yoy")
    with trend_cols[2]:
        chart(yoy_metric_figure(monthly, "AOV", "Monthly AOV by year"), "overview-aov-yoy")
    st.caption("Each line represents a calendar year so the same month can be compared across years. Partial years stop at the last observed month.")

    st.subheader("Pareto analysis")
    metric = _centered_pills("Pareto measure", ("Orders", "GMV", "AOV"), "overview-contribution-measure", "Orders")
    st.caption("Use the same business metric to compare concentration across customer geography and products. Orders and GMV include cumulative Pareto share; AOV is shown as a ranked comparison because it is non-additive.")

    geography_slot = st.empty()
    geography_level = _centered_pills("Customer geography level", ("Region", "State"), "overview-geography-level", "Region")
    geography_column = "customer_region" if geography_level == "Region" else "customer_state"
    with geography_slot.container():
        geo_fig = contribution_figure(frame, geography_column, metric, f"{metric} by customer {geography_level.lower()}")
        chart(geo_fig, "overview-geography-contribution") if geo_fig.data else st.info("No customer geography matches the current filters.")

    category_slot = st.empty()
    category_level = _centered_pills("Product category level", ("Category group", "Category"), "overview-category-level", "Category group")
    category_column = "product_group" if category_level == "Category group" else "product_category"
    with category_slot.container():
        cat_fig = contribution_figure(frame, category_column, metric, f"{metric} by {category_level.lower()}")
        chart(cat_fig, "overview-category-contribution") if cat_fig.data else st.info("No product category matches the current filters.")

    st.caption("Top contributors remain visible and the long tail is combined into Others where appropriate.")


def region_late_heatmap(valid: pd.DataFrame) -> None:
    st.subheader("Late delivery by route region")
    st.caption("Compare late-delivery rates across seller and customer regions to identify where missed promises are concentrated.")
    minimum_orders = st.slider("Minimum eligible orders per region pair", 1, 2000, 100, 25, key="promise-region-minimum", help="Hide region pairs with too few eligible delivered orders to support a stable descriptive rate.")
    regional = valid.dropna(subset=["customer_region", "seller_region"]).groupby(["customer_region", "seller_region"], as_index=False).agg(eligible_orders=("order_id", "nunique"), late_rate=("is_late", "mean"))
    regional = regional[regional["eligible_orders"] >= minimum_orders]
    if regional.empty:
        st.info("No seller/customer region pair meets the current minimum-order threshold.")
        return
    late = regional.pivot(index="customer_region", columns="seller_region", values="late_rate").reindex(index=REGION_ORDER, columns=REGION_ORDER)
    counts = regional.pivot(index="customer_region", columns="seller_region", values="eligible_orders").reindex(index=REGION_ORDER, columns=REGION_ORDER)
    fig = go.Figure(go.Heatmap(z=late.values, x=late.columns.tolist(), y=late.index.tolist(), colorscale="Reds", zmin=0, customdata=counts.values, texttemplate="%{z:.1%}", hovertemplate="Seller region: %{x}<br>Customer region: %{y}<br>Late delivery rate: %{z:.1%}<br>Eligible orders: %{customdata:,.0f}<extra></extra>", colorbar={"title": "Late %", "tickformat": ".0%"}))
    fig.update_layout(title="Late delivery rate by seller region × customer region", xaxis_title="Seller region", yaxis_title="Customer region")
    chart(styled(fig, 520), "promise-region-heatmap")
    st.caption("Darker red means a higher share of delivered orders arrived after the promised date. This is descriptive, not causal.")


def delivery_promise(frame: pd.DataFrame) -> None:
    st.header("Delivery Analysis")
    st.caption("Assess how actual delivery compares with the promised date, how orders are distributed from early to late, and where late delivery is concentrated geographically.")
    valid = delivered_orders(frame)
    if valid.empty:
        st.info("No delivered orders with valid quote and delivery timestamps match the filters.")
        return

    valid["days_from_promise"] = valid["days_late"].astype(float)
    median_actual = valid["delivery_days"].median()
    median_quote = valid["estimated_delivery_days"].median()
    median_gap = valid["days_from_promise"].median()
    quote_mae = valid["days_from_promise"].abs().mean()
    page_insight(
        f"Median actual delivery is {median_actual:,.0f} days versus {median_quote:,.0f} promised days, and the median order arrived {abs(median_gap):,.0f} days {'late' if median_gap > 0 else 'early'}."
    )
    cols = st.columns(4)
    cols[0].metric("Median actual delivery", f"{number(median_actual, 0)} days")
    cols[1].metric("Median promised delivery", f"{number(median_quote, 0)} days")
    cols[2].metric("Median vs promise", f"{number(abs(median_gap), 0)}d {'late' if median_gap > 0 else 'early'}", help="Difference between actual and promised delivery timing, summarised by the median order.")
    cols[3].metric("Promise MAE", f"{number(quote_mae, 1)} days", help="MAE = Mean Absolute Error. It is the average absolute gap between actual and promised delivery timing; lower means the promise is closer to reality.")

    left, right = st.columns(2)
    with left:
        quote = (
            valid.groupby("estimated_delivery_days", as_index=False)
            .agg(**{"Median actual delivery days": ("delivery_days", "median"), "Orders": ("order_id", "nunique")})
            .sort_values("estimated_delivery_days")
        )
        stable_quote = quote[quote["Orders"] >= 25].copy()
        if stable_quote.empty:
            stable_quote = quote.copy()
        axis_max = max(float(valid["estimated_delivery_days"].quantile(0.995)), float(valid["delivery_days"].quantile(0.995)), 1.0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valid["estimated_delivery_days"], y=valid["delivery_days"], mode="markers", name="Delivered orders",
            marker={"color": BLUE, "size": 5, "opacity": 0.12},
            hovertemplate="Promised: %{x:.0f} days<br>Actual: %{y:.0f} days<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=stable_quote["estimated_delivery_days"], y=stable_quote["Median actual delivery days"], mode="lines+markers",
            name="Median actual delivery", customdata=stable_quote[["Orders"]], line={"color": ORANGE, "width": 3},
            hovertemplate="Promised: %{x:.0f} days<br>Median actual: %{y:.1f} days<br>Orders in band: %{customdata[0]:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(x=[0, axis_max], y=[0, axis_max], mode="lines", name="Actual = promised", line={"color": GREY, "dash": "dash", "width": 2}, hoverinfo="skip"))
        fig.update_layout(
            title="Actual vs promised delivery",
            xaxis={"title": "Promised delivery days", "range": [0, axis_max * 1.03]},
            yaxis={"title": "Actual delivery days", "range": [0, axis_max * 1.03]},
            legend={"orientation": "h", "y": 1.18, "x": 0},
        )
        chart(styled(fig, 470), "promise-quoted-actual")
        st.caption("Each point is a delivered order. The orange line shows median actual delivery for sufficiently populated promise-day values; the dashed line marks actual = promised.")

    with right:
        bins = [-999, -30, -15, -7, -3, 0, 3, 7, 15, 30, 999]
        labels = ["30+ early", "15–30 early", "7–15 early", "3–7 early", "0–3 early", "0–3 late", "3–7 late", "7–15 late", "15–30 late", "30+ late"]
        valid["Delivery timing"] = pd.cut(valid["days_from_promise"], bins=bins, labels=labels, ordered=True)
        counts = valid.dropna(subset=["Delivery timing"]).groupby("Delivery timing", observed=False)["order_id"].nunique().rename("Orders").reset_index()
        counts["Delivery timing"] = counts["Delivery timing"].astype(str)
        early = counts[counts["Delivery timing"].str.contains("early")]
        late = counts[counts["Delivery timing"].str.contains("late")]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=early["Delivery timing"], y=early["Orders"], name="Early", marker_color=GREEN,
            text=early["Orders"], texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False,
            customdata=early[["Orders"]], hovertemplate="%{x}<br>Orders: %{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=late["Delivery timing"], y=late["Orders"], name="Late", marker_color=RED,
            text=late["Orders"], texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False,
            customdata=late[["Orders"]], hovertemplate="%{x}<br>Orders: %{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(
            title="Orders by timing relative to promise",
            xaxis={"title": "Days from promise (early ← promise → late)", "categoryorder": "array", "categoryarray": labels, "tickangle": -35},
            yaxis={"title": "Orders", "tickformat": ","},
            barmode="group",
            legend={"orientation": "h", "y": 1.16, "x": 0},
        )
        chart(styled(fig, 470), "promise-timing-counts")
        st.caption("Green bars are early deliveries and red bars are late deliveries; bar height shows the number of orders in each timing band.")

    region_late_heatmap(valid)
    st.caption(f"Delivery analysis uses {valid['order_id'].nunique():,} eligible completed delivered orders under the current filters.")


def delivery_experience(frame: pd.DataFrame) -> None:
    st.header("Negative Review Analysis")
    st.caption("Assess how delivery timing relates to customer review outcomes, focusing on average review score and negative review rate.")
    reviewed = delivered_orders(frame).dropna(subset=["review_score", "is_negative_review"]).copy()
    if reviewed.empty:
        st.info("No reviewed orders with valid delivery outcomes match the filters.")
        return
    reviewed["days_early"] = -reviewed["days_late"].astype(float)
    bins = [-999, -30, -15, -7, -3, 0, 3, 7, 15, 999]
    labels = ["30+ late", "15–30 late", "7–15 late", "3–7 late", "0–3 late", "0–3 early", "3–7 early", "7–15 early", "15+ early"]
    reviewed["Delivery timing"] = pd.cut(reviewed["days_early"], bins=bins, labels=labels, ordered=True)
    timing = reviewed.dropna(subset=["Delivery timing"]).groupby("Delivery timing", observed=False, as_index=False).agg(**{"Average review score": ("review_score", "mean"), "Negative review rate": ("is_negative_review", "mean"), "Reviewed orders": ("order_id", "nunique")})
    timing["Delivery timing"] = pd.Categorical(timing["Delivery timing"].astype(str), categories=labels, ordered=True)
    timing = timing.sort_values("Delivery timing", ascending=False)
    page_insight("Average review scores fall and negative review rates rise as deliveries move further past the promised date. This is an exploratory association, not evidence of causation.")
    cols = st.columns(3)
    cols[0].metric("Reviewed delivered orders", f"{reviewed['order_id'].nunique():,}")
    cols[1].metric("Average review score", number(reviewed["review_score"].mean(), 2))
    cols[2].metric("Negative review rate", percent(reviewed["is_negative_review"].mean()), help="Share of reviewed delivered orders classified as a negative review under the Phase 1 prepared-data definition.")

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(
            x=timing["Delivery timing"].astype(str), y=timing["Average review score"], name="Average review score",
            text=timing["Average review score"], texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False,
            marker={"color": timing["Average review score"], "colorscale": "RdYlGn", "cmin": 1, "cmax": 5, "showscale": True, "colorbar": {"title": "Review score<br>Red = worse<br>Green = better"}},
            customdata=timing[["Reviewed orders", "Negative review rate"]],
            hovertemplate="%{x}<br>Average review score: %{y:.2f}<br>Reviewed orders: %{customdata[0]:,.0f}<br>Negative review rate: %{customdata[1]:.1%}<extra></extra>",
        ))
        fig.update_layout(title="Average review score by delivery timing", showlegend=True, legend={"orientation": "h", "y": 1.16, "x": 0})
        fig.update_yaxes(title="Average review score (1–5)", range=[1, 5.4])
        fig.update_xaxes(title="Delivery timing relative to promise", tickangle=-35)
        chart(styled(fig, 470), "experience-review-score")

    with right:
        fig = go.Figure(go.Bar(
            x=timing["Delivery timing"].astype(str), y=timing["Negative review rate"], name="Negative review rate",
            text=timing["Negative review rate"], texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False,
            marker={"color": timing["Negative review rate"], "colorscale": "RdYlGn_r", "cmin": 0, "cmax": 1, "showscale": True, "colorbar": {"title": "Negative review rate<br>Green = better<br>Red = worse", "tickformat": ".0%"}},
            customdata=timing[["Reviewed orders", "Average review score"]],
            hovertemplate="%{x}<br>Negative review rate: %{y:.1%}<br>Reviewed orders: %{customdata[0]:,.0f}<br>Average review score: %{customdata[1]:.2f}<extra></extra>",
        ))
        fig.update_layout(title="Negative review rate by delivery timing", showlegend=True, legend={"orientation": "h", "y": 1.16, "x": 0})
        fig.update_yaxes(title="Negative review rate", tickformat=".0%", range=[0, max(float(timing["Negative review rate"].max()) * 1.15, 0.1)])
        fig.update_xaxes(title="Delivery timing relative to promise", tickangle=-35)
        chart(styled(fig, 470), "experience-negative-review-rate")

    st.caption("Both charts use the same early-to-late timing order. Colour has the same meaning in both: green = better customer outcome, red = worse customer outcome.")


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
    return date_range, customer_regions, customer_states, customer_cities, seller_regions, seller_states, route_types, product_groups, categories, status_groups, statuses


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
        st.caption(f"**Analysis window:** {pd.Timestamp(date_range[0]).strftime('%b %Y')} to {pd.Timestamp(date_range[1]).strftime('%b %Y')} · **Selected:** {filtered['order_id'].nunique():,} of {data['order_id'].nunique():,} orders")
    if filtered.empty:
        st.warning("No orders match the current filter combination.")
        return
    overview_tab, promise_tab, experience_tab = st.tabs(["Overview", "Delivery Analysis", "Negative Review Analysis"])
    with overview_tab: business_overview(filtered)
    with promise_tab: delivery_promise(filtered)
    with experience_tab: delivery_experience(filtered)


if __name__ == "__main__":
    main()