"""Reusable review-score correlation helpers for the Streamlit dashboard."""

import pandas as pd
import plotly.graph_objects as go

RED = "#DC2626"
GREEN = "#0F766E"
GREY = "#64748B"


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
    seen_labels = set()
    for column, label in candidates.items():
        if column not in data.columns or label in seen_labels:
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
        seen_labels.add(label)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["Absolute r"] = result["Pearson r"].abs()
    return result.sort_values("Absolute r", ascending=True).drop(columns="Absolute r")


def correlation_figure(correlations: pd.DataFrame) -> go.Figure:
    """Show correlations with review score as a compact horizontal bar chart."""
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
