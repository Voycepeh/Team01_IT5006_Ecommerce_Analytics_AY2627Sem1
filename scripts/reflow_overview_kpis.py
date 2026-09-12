from pathlib import Path

path = Path("deployment/streamlit/app.py")
text = path.read_text(encoding="utf-8")

old = '''    for metric in ("Orders", "GMV", "AOV"):
        st.subheader(metric)
        kpi_col, chart_col = st.columns([1, 2])
        with kpi_col:
            total_value = _overview_metric_value(frame, metric)
            kpi_col.metric(metric_config[metric]["total_label"], _overview_metric_text(total_value, metric), help=metric_config[metric]["help"], border=True)
            year_cols = st.columns(2)
            for year_col, year in zip(year_cols, (2017, 2018)):
                year_frame = frame[frame["purchase_date"].dt.year == year]
                year_value = _overview_metric_value(year_frame, metric)
                year_col.metric(str(year), _overview_metric_text(year_value, metric), border=True)
        with chart_col:
            trend_view = _centered_pills(
                f"{metric} trend view",
                ("All-time trend", "Year-on-year"),
                f"overview-{metric.lower()}-trend-view",
                "All-time trend",
            )
            figure = (
                all_time_metric_figure(monthly, metric, f"Monthly {metric} over time")
                if trend_view == "All-time trend"
                else yoy_metric_figure(monthly, metric, f"Monthly {metric} by year")
            )
            chart(figure, f"overview-{metric.lower()}-yoy")
        st.divider()
'''

new = '''    for metric in ("Orders", "GMV", "AOV"):
        st.subheader(metric)

        kpi_cols = st.columns(3)
        total_value = _overview_metric_value(frame, metric)
        kpi_cols[0].metric(
            metric_config[metric]["total_label"],
            _overview_metric_text(total_value, metric),
            help=metric_config[metric]["help"],
            border=True,
        )
        for year_col, year in zip(kpi_cols[1:], (2017, 2018)):
            year_frame = frame[frame["purchase_date"].dt.year == year]
            year_value = _overview_metric_value(year_frame, metric)
            year_col.metric(str(year), _overview_metric_text(year_value, metric), border=True)

        trend_view = _centered_pills(
            f"{metric} trend view",
            ("All-time trend", "Year-on-year"),
            f"overview-{metric.lower()}-trend-view",
            "All-time trend",
        )
        figure = (
            all_time_metric_figure(monthly, metric, f"Monthly {metric} over time")
            if trend_view == "All-time trend"
            else yoy_metric_figure(monthly, metric, f"Monthly {metric} by year")
        )
        chart(figure, f"overview-{metric.lower()}-yoy")
        st.divider()
'''

if old not in text:
    raise RuntimeError("Overview metric layout block not found")
text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
