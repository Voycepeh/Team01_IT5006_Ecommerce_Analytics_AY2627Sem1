from pathlib import Path

path = Path("deployment/streamlit/app.py")
text = path.read_text(encoding="utf-8")


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match, found {count}: {old[:120]!r}")
    return source.replace(old, new, 1)


helper_anchor = "\n\ndef region_late_heatmap(valid: pd.DataFrame) -> None:\n"
helper_block = '''

def delivery_timing_distribution_figure(valid: pd.DataFrame) -> go.Figure:
    """Show the full distribution of delivery timing around the promised date."""
    distribution = valid["days_from_promise"].dropna().astype(float)
    fig = go.Figure(go.Histogram(
        x=distribution,
        nbinsx=min(max(int(distribution.nunique()), 20), 120),
        marker_color=BLUE,
        opacity=0.9,
        hovertemplate="Days from promise: %{x}<br>Orders: %{y:,.0f}<extra></extra>",
        showlegend=False,
    ))
    median_gap = float(distribution.median())
    lower = float(distribution.quantile(0.005))
    upper = float(distribution.quantile(0.995))
    if lower == upper:
        lower -= 1.0
        upper += 1.0
    padding = max((upper - lower) * 0.08, 5.0)
    fig.add_vline(x=0, line_color=ORANGE, line_width=2)
    fig.add_vline(x=median_gap, line_color=GREEN, line_width=2, line_dash="dash")
    fig.update_layout(
        title="How far deliveries land from the promised date",
        xaxis={
            "title": "Days from promised date (negative = early, 0 = on time, positive = late)",
            "range": [lower - padding, upper + padding],
        },
        yaxis={"title": "Orders", "tickformat": ",", "rangemode": "tozero"},
        bargap=0.02,
    )
    return styled(fig, 470)


def delivery_stage_breakdown_figure(valid: pd.DataFrame) -> go.Figure:
    """Compare median handling and transit days for on-time-or-early versus late deliveries."""
    required_columns = {
        "purchase_timestamp",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "days_from_promise",
        "order_id",
    }
    if not required_columns.issubset(valid.columns):
        return go.Figure()

    stage_frame = valid[list(required_columns)].copy()
    for column in ("purchase_timestamp", "order_delivered_carrier_date", "order_delivered_customer_date"):
        stage_frame[column] = pd.to_datetime(stage_frame[column], errors="coerce")
    stage_frame = stage_frame.dropna(
        subset=[
            "purchase_timestamp",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "days_from_promise",
        ]
    ).copy()
    if stage_frame.empty:
        return go.Figure()

    stage_frame["handling_days"] = (
        stage_frame["order_delivered_carrier_date"] - stage_frame["purchase_timestamp"]
    ).dt.total_seconds() / 86400
    stage_frame["transit_days"] = (
        stage_frame["order_delivered_customer_date"] - stage_frame["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86400
    stage_frame = stage_frame[
        (stage_frame["handling_days"] >= 0) & (stage_frame["transit_days"] >= 0)
    ].copy()
    if stage_frame.empty:
        return go.Figure()

    stage_frame["Delivery outcome"] = stage_frame["days_from_promise"].astype(float).le(0).map(
        {True: "On time or early", False: "Late"}
    )
    summary = stage_frame.groupby("Delivery outcome", as_index=False).agg(
        **{
            "Handling days": ("handling_days", "median"),
            "Transit days": ("transit_days", "median"),
            "Orders": ("order_id", "nunique"),
        }
    )
    outcome_order = ["Late", "On time or early"]
    summary["Delivery outcome"] = pd.Categorical(
        summary["Delivery outcome"], categories=outcome_order, ordered=True
    )
    summary = summary.sort_values("Delivery outcome")
    if summary.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=summary["Delivery outcome"],
        x=summary["Handling days"],
        name="Handling days",
        orientation="h",
        marker_color=BLUE,
        customdata=summary[["Orders"]],
        hovertemplate="%{y}<br>Median handling days: %{x:.1f}<br>Orders: %{customdata[0]:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=summary["Delivery outcome"],
        x=summary["Transit days"],
        name="Transit days",
        orientation="h",
        marker_color=ORANGE,
        customdata=summary[["Orders"]],
        hovertemplate="%{y}<br>Median transit days: %{x:.1f}<br>Orders: %{customdata[0]:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Where delivery time goes: handling vs transit",
        xaxis={"title": "Median days", "rangemode": "tozero"},
        yaxis={"title": ""},
        barmode="stack",
        legend={"orientation": "h", "y": 1.14, "x": 0.5, "xanchor": "center"},
    )
    return styled(fig, 470)
'''

if "def delivery_timing_distribution_figure(valid: pd.DataFrame) -> go.Figure:" not in text:
    if text.count(helper_anchor) != 1:
        raise RuntimeError("Could not find region_late_heatmap insertion point")
    text = text.replace(helper_anchor, helper_block + helper_anchor, 1)

row_old = '''        st.caption("Green bars are early deliveries, blue is exactly on the promised date (0 days), and red bars are late deliveries; bar height shows the number of orders in each timing band.")

    region_late_heatmap(valid)'''
row_new = '''        st.caption("Green bars are early deliveries, blue is exactly on the promised date (0 days), and red bars are late deliveries; bar height shows the number of orders in each timing band.")

    detail_left, detail_right = st.columns(2)
    with detail_left:
        chart(delivery_timing_distribution_figure(valid), "promise-timing-distribution")
        st.caption("This histogram shows the full spread of delivery timing around the promised date. The orange reference line marks the promised date itself (0 days) and the green dashed line marks the median delivered order.")

    with detail_right:
        stage_figure = delivery_stage_breakdown_figure(valid)
        if stage_figure.data:
            chart(stage_figure, "promise-stage-breakdown")
            st.caption("Handling days run from purchase to carrier handoff, while transit days run from carrier handoff to customer delivery. Bars show median stage lengths for on-time-or-early versus late deliveries.")
        else:
            st.info("Handling-versus-transit breakdown is unavailable because the required delivery-stage timestamps are missing for the current filters.")

    region_late_heatmap(valid)'''

if "promise-timing-distribution" not in text:
    text = replace_once(text, row_old, row_new)

path.write_text(text, encoding="utf-8")

test_path = Path("tests/test_streamlit_dashboard.py")
test_text = test_path.read_text(encoding="utf-8")
test_text = test_text.replace(
    'assert {trace.name for trace in timing_fig.data} == {"Early", "Late"}',
    'assert {trace.name for trace in timing_fig.data} == {"Early", "On time", "Late"}',
)
if '"promise-timing-distribution", "promise-region-heatmap"' not in test_text:
    test_text = test_text.replace(
        '"promise-quoted-actual", "promise-timing-counts", "promise-region-heatmap",',
        '"promise-quoted-actual", "promise-timing-counts", "promise-timing-distribution", "promise-region-heatmap",',
    )
test_path.write_text(test_text, encoding="utf-8")
