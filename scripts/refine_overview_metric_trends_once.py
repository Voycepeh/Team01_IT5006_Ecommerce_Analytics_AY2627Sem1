from pathlib import Path

path = Path('deployment/streamlit/app.py')
text = path.read_text(encoding='utf-8')

# Remove the duplicated combined Business trend helper.
start = text.index('\ndef overview_trend_figure(')
end = text.index('\ndef contribution_figure(', start)
text = text[:start] + '\n' + text[end:]

# Add a single-metric all-time helper before contribution_figure.
insert_at = text.index('\ndef contribution_figure(')
helper = '''\ndef all_time_metric_figure(monthly: pd.DataFrame, metric: str, title: str) -> go.Figure:\n    y_title = "Orders" if metric == "Orders" else ("Gross merchandise value (R$)" if metric == "GMV" else "Average order value (R$)")\n    hover = "%{x|%b %Y}<br>Orders: %{y:,.0f}<extra></extra>" if metric == "Orders" else f"%{{x|%b %Y}}<br>{metric}: R$%{{y:,.2f}}<extra></extra>"\n    fig = go.Figure(go.Scatter(\n        x=monthly["Month"], y=monthly[metric], mode="lines+markers",\n        line={"width": 2.5, "color": BLUE}, marker={"size": 6},\n        hovertemplate=hover, showlegend=False,\n    ))\n    fig.update_layout(\n        title=title,\n        xaxis={"title": "Purchase month"},\n        yaxis={"title": y_title, "tickformat": ",", "rangemode": "tozero"},\n    )\n    if metric in {"GMV", "AOV"}:\n        fig.update_yaxes(tickprefix="R$")\n    return styled(fig, 360)\n\n'''
text = text[:insert_at] + helper + text[insert_at:]

# Remove the duplicated top Business trend block.
old = '''    st.subheader("Business trend")\n    trend_view = _centered_pills("Trend view", ("All-time trend", "Year-on-year"), "overview-trend-view", "All-time trend")\n    chart(overview_trend_figure(monthly, trend_view), "overview-business-trend")\n    st.caption("All-time trend shows the full monthly trajectory; Year-on-year aligns calendar months for comparison. Both y-axes start at zero so small variations are not visually exaggerated.")\n    st.divider()\n\n'''
if old not in text:
    raise RuntimeError('Business trend block not found')
text = text.replace(old, '', 1)

# Give each existing metric chart its own view toggle.
old = '''        with chart_col:\n            chart(yoy_metric_figure(monthly, metric, f"Monthly {metric} by year"), f"overview-{metric.lower()}-yoy")\n        st.divider()\n\n    st.caption("Each line represents a calendar year so the same month can be compared across years. Partial years stop at the last observed month.")\n'''
new = '''        with chart_col:\n            trend_view = _centered_pills(\n                f"{metric} trend view",\n                ("All-time trend", "Year-on-year"),\n                f"overview-{metric.lower()}-trend-view",\n                "All-time trend",\n            )\n            figure = (\n                all_time_metric_figure(monthly, metric, f"Monthly {metric} over time")\n                if trend_view == "All-time trend"\n                else yoy_metric_figure(monthly, metric, f"Monthly {metric} by year")\n            )\n            chart(figure, f"overview-{metric.lower()}-yoy")\n        st.divider()\n\n    st.caption("Use All-time trend for the full monthly trajectory or Year-on-year to align calendar months across years. All y-axes start at zero.")\n'''
if old not in text:
    raise RuntimeError('Metric chart block not found')
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
