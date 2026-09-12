from pathlib import Path

path = Path("deployment/streamlit/app.py")
text = path.read_text(encoding="utf-8")

# Remove the duplicated combined Business trend helper.
start = text.index("\ndef overview_trend_figure(monthly: pd.DataFrame, view: str) -> go.Figure:\n")
end = text.index("\ndef contribution_figure(frame: pd.DataFrame, dimension: str, metric: str, title: str) -> go.Figure:\n", start)
text = text[:start] + text[end:]

# Add a reusable all-time trend helper immediately after yoy_metric_figure.
anchor = "    return styled(fig, 360)\n\n\n\ndef contribution_figure"
replacement = '''    return styled(fig, 360)\n\n\ndef all_time_metric_figure(monthly: pd.DataFrame, metric: str, title: str) -> go.Figure:\n    \"\"\"Show one business metric across the full monthly analysis window.\"\"\"\n    if metric == \"Orders\":\n        y_title = \"Orders\"\n        hover = \"%{x|%b %Y}<br>Orders: %{y:,.0f}<extra></extra>\"\n    elif metric == \"GMV\":\n        y_title = \"Gross merchandise value (R$)\"\n        hover = \"%{x|%b %Y}<br>GMV: R$%{y:,.2f}<extra></extra>\"\n    else:\n        y_title = \"Average order value (R$)\"\n        hover = \"%{x|%b %Y}<br>AOV: R$%{y:,.2f}<extra></extra>\"\n\n    fig = go.Figure(go.Scatter(\n        x=monthly[\"Month\"],\n        y=monthly[metric],\n        mode=\"lines+markers\",\n        name=metric,\n        line={\"width\": 2.5, \"color\": BLUE},\n        marker={\"size\": 6},\n        hovertemplate=hover,\n    ))\n    fig.update_layout(\n        title=title,\n        xaxis={\"title\": \"Purchase month\"},\n        yaxis={\"title\": y_title, \"tickformat\": \",\", \"rangemode\": \"tozero\"},\n        showlegend=False,\n    )\n    if metric in {\"GMV\", \"AOV\"}:\n        fig.update_yaxes(tickprefix=\"R$\")\n    return styled(fig, 360)\n\n\ndef contribution_figure'''
if anchor not in text:
    raise RuntimeError("Could not find insertion anchor for all-time metric helper")
text = text.replace(anchor, replacement, 1)

# Remove the duplicated combined Business trend section from Overview.
business_block = '''    st.subheader("Business trend")\n    trend_view = _centered_pills("Trend view", ("All-time trend", "Year-on-year"), "overview-trend-view", "All-time trend")\n    chart(overview_trend_figure(monthly, trend_view), "overview-business-trend")\n    st.caption("All-time trend shows the full monthly trajectory; Year-on-year aligns calendar months for comparison. Both y-axes start at zero so small variations are not visually exaggerated.")\n    st.divider()\n\n'''
if business_block not in text:
    raise RuntimeError("Could not find duplicated Business trend block")
text = text.replace(business_block, "", 1)

# Give each Orders/GMV/AOV row its own all-time vs YoY toggle while preserving KPI layout.
old = '''        with chart_col:\n            chart(yoy_metric_figure(monthly, metric, f"Monthly {metric} by year"), f"overview-{metric.lower()}-yoy")\n        st.divider()\n\n    st.caption("Each line represents a calendar year so the same month can be compared across years. Partial years stop at the last observed month.")\n'''
new = '''        with chart_col:\n            metric_view = _centered_pills(\n                f"{metric} trend view",\n                ("All-time trend", "Year-on-year"),\n                f"overview-{metric.lower()}-trend-view",\n                "All-time trend",\n            )\n            if metric_view == "All-time trend":\n                metric_figure = all_time_metric_figure(monthly, metric, f"Monthly {metric} over time")\n            else:\n                metric_figure = yoy_metric_figure(monthly, metric, f"Monthly {metric} by year")\n            chart(metric_figure, f"overview-{metric.lower()}-yoy")\n        st.divider()\n\n    st.caption("Use All-time trend to follow the full monthly trajectory, or Year-on-year to compare the same calendar month across years. All trend y-axes start at zero.")\n'''
if old not in text:
    raise RuntimeError("Could not find Overview metric chart block")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
