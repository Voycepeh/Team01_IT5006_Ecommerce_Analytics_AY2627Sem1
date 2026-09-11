from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


APP_PATH = Path(__file__).resolve().parents[1] / "deployment" / "streamlit" / "app.py"
spec = importlib.util.spec_from_file_location("olist_dashboard", APP_PATH)
app = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(app)


class _FakeColumn:
    def metric(self, *args, **kwargs):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeStreamlit:
    def header(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def columns(self, count):
        return [_FakeColumn() for _ in range(count)]

    def radio(self, label, options, **kwargs):
        return options[0]

    def slider(self, *args, **kwargs):
        return 1


def _frame() -> pd.DataFrame:
    rows = [
        dict(order_id="A1", customer_unique_id="U1", purchase_date="2017-03-10", order_status="delivered", customer_state="SP", customer_city="sao paulo", seller_state="SP", product_category="books_general_interest", total_item_value=100.0, review_score=5.0, delivery_days=6.0, estimated_delivery_days=14.0, days_late=-8.0, is_delivered_complete=True, is_late=False, is_negative_review=False),
        dict(order_id="A2", customer_unique_id="U2", purchase_date="2017-04-10", order_status="delivered", customer_state="SP", customer_city="sao paulo", seller_state="SP", product_category="books_general_interest", total_item_value=120.0, review_score=4.0, delivery_days=8.0, estimated_delivery_days=12.0, days_late=-4.0, is_delivered_complete=True, is_late=False, is_negative_review=False),
        dict(order_id="B1", customer_unique_id="U3", purchase_date="2017-03-11", order_status="delivered", customer_state="RJ", customer_city="rio de janeiro", seller_state="SP", product_category="toys", total_item_value=200.0, review_score=2.0, delivery_days=16.0, estimated_delivery_days=12.0, days_late=4.0, is_delivered_complete=True, is_late=True, is_negative_review=True),
        dict(order_id="C1", customer_unique_id="U4", purchase_date="2018-02-15", order_status="delivered", customer_state="PR", customer_city="curitiba", seller_state="MG", product_category="health_beauty", total_item_value=80.0, review_score=3.0, delivery_days=14.0, estimated_delivery_days=13.0, days_late=1.0, is_delivered_complete=True, is_late=True, is_negative_review=False),
        dict(order_id="D1", customer_unique_id="U5", purchase_date="2017-03-12", order_status="canceled", customer_state="SP", customer_city="sao paulo", seller_state="SP", product_category="books_general_interest", total_item_value=90.0, review_score=None, delivery_days=None, estimated_delivery_days=10.0, days_late=None, is_delivered_complete=False, is_late=pd.NA, is_negative_review=pd.NA),
    ]
    frame = pd.DataFrame(rows)
    frame["purchase_date"] = pd.to_datetime(frame["purchase_date"])
    frame["is_delivered_complete"] = frame["is_delivered_complete"].astype(bool)
    frame["is_late"] = frame["is_late"].astype("boolean")
    frame["is_negative_review"] = frame["is_negative_review"].astype("boolean")
    return app.enrich_dimensions(frame)


def _apply(frame, **overrides):
    args = dict(
        date_range=(pd.Timestamp("2017-01-01").date(), pd.Timestamp("2018-08-31").date()),
        customer_regions=[],
        customer_states=[],
        customer_cities=[],
        seller_regions=[],
        seller_states=[],
        route_types=[],
        product_groups=[],
        categories=[],
        status_groups=[],
        statuses=[],
    )
    args.update(overrides)
    return app.apply_filters(frame, **args)


def test_every_filter_dimension_individually():
    frame = _frame()
    assert set(_apply(frame, date_range=(pd.Timestamp("2018-01-01").date(), pd.Timestamp("2018-12-31").date()))["order_id"]) == {"C1"}
    assert set(_apply(frame, customer_regions=["South"])["order_id"]) == {"C1"}
    assert set(_apply(frame, customer_states=["RJ"])["order_id"]) == {"B1"}
    assert set(_apply(frame, customer_cities=["curitiba"])["order_id"]) == {"C1"}
    assert set(_apply(frame, seller_regions=["Southeast"])["order_id"]) == {"A1", "A2", "B1", "C1", "D1"}
    assert set(_apply(frame, seller_states=["MG"])["order_id"]) == {"C1"}
    assert set(_apply(frame, route_types=["Cross state"])["order_id"]) == {"B1", "C1"}
    assert set(_apply(frame, product_groups=["Health & Beauty"])["order_id"]) == {"C1"}
    assert set(_apply(frame, categories=["toys"])["order_id"]) == {"B1"}
    assert set(_apply(frame, status_groups=["Exception"])["order_id"]) == {"D1"}
    assert set(_apply(frame, statuses=["canceled"])["order_id"]) == {"D1"}


def test_parent_groupings_are_reproducible():
    frame = _frame()
    assert set(frame.loc[frame["customer_region"] == "Southeast", "customer_state"]) == {"SP", "RJ"}
    assert set(frame.loc[frame["product_group"] == "Books, Media & Stationery", "product_category"]) == {"books_general_interest"}
    assert set(frame.loc[frame["status_group"] == "Fulfilled", "order_status"]) == {"delivered"}
    assert set(app._child_options(frame, "customer_region", ["South"], "customer_state")) == {"PR"}
    assert set(app._child_options(frame, "product_group", ["Health & Beauty"], "product_category")) == {"health_beauty"}
    assert set(app._child_options(frame, "status_group", ["Exception"], "order_status")) == {"canceled"}


def test_multiple_parent_and_child_filters_are_anded_together():
    frame = _frame()
    filtered = _apply(
        frame,
        date_range=(pd.Timestamp("2017-03-01").date(), pd.Timestamp("2017-04-30").date()),
        customer_regions=["Southeast"],
        customer_states=["SP"],
        customer_cities=["sao paulo"],
        seller_regions=["Southeast"],
        seller_states=["SP"],
        route_types=["Same state"],
        product_groups=["Books, Media & Stationery"],
        categories=["books_general_interest"],
        status_groups=["Fulfilled"],
        statuses=["delivered"],
    )
    assert list(filtered["order_id"]) == ["A1", "A2"]


def test_all_visuals_use_the_same_multi_filtered_population(monkeypatch):
    frame = _frame()
    filtered = _apply(
        frame,
        date_range=(pd.Timestamp("2017-03-01").date(), pd.Timestamp("2017-04-30").date()),
        customer_regions=["Southeast"], customer_states=["SP"], customer_cities=["sao paulo"],
        seller_regions=["Southeast"], seller_states=["SP"], route_types=["Same state"],
        product_groups=["Books, Media & Stationery"], categories=["books_general_interest"],
        status_groups=["Fulfilled"], statuses=["delivered"],
    )
    assert set(filtered["order_id"]) == {"A1", "A2"}

    captured = []
    monkeypatch.setattr(app, "st", _FakeStreamlit())
    monkeypatch.setattr(app, "page_insight", lambda text: None)
    monkeypatch.setattr(app, "chart", lambda figure, key: captured.append((key, figure)))

    app.business_overview(filtered)
    app.delivery_promise(filtered)
    app.delivery_experience(filtered)

    figures = {key: figure for key, figure in captured}
    expected_keys = {
        "overview-indexed-growth", "overview-orders", "overview-aov", "overview-pareto",
        "promise-quoted-actual", "promise-timing-counts", "experience-timing",
        "experience-route-days", "experience-route-late", "experience-route-heatmap",
    }
    assert expected_keys.issubset(figures), expected_keys.difference(figures)
    assert sum(figures["overview-orders"].data[0].y) == 2
    pareto_bar = figures["overview-pareto"].data[0]
    assert list(pareto_bar.x) == ["books_general_interest"]
    assert sum(pareto_bar.y) == 2
    quote_fig = figures["promise-quoted-actual"]
    quote_orders = sum(trace.customdata[:, 0].astype(float).sum() for trace in quote_fig.data if getattr(trace, "customdata", None) is not None)
    assert quote_orders == 2
    assert sum(figures["promise-timing-counts"].data[0].y) == 2
    experience_bar = figures["experience-timing"].data[0]
    assert sum(experience_bar.text) == 2
    assert set(figures["experience-route-days"].data[0].x) == {"Same state"}
    assert set(figures["experience-route-late"].data[0].x) == {"Same state"}
    heatmap = figures["experience-route-heatmap"].data[0]
    assert list(heatmap.x) == ["SP"]
    assert list(heatmap.y) == ["SP"]
    assert float(heatmap.customdata[0][0]) == 2


def test_pareto_top_10_plus_others_uses_full_population(monkeypatch):
    frame = _frame()
    extra = []
    for i in range(12):
        extra.append(dict(
            order_id=f"P{i}", customer_unique_id=f"PU{i}", purchase_date=pd.Timestamp("2017-05-01"),
            order_status="delivered", customer_state="SP", customer_city="sao paulo", seller_state="SP",
            product_category=f"cat_{i:02d}", total_item_value=float(200 - i * 5), review_score=5.0,
            delivery_days=5.0, estimated_delivery_days=10.0, days_late=-5.0,
            is_delivered_complete=True, is_late=False, is_negative_review=False,
        ))
    frame = app.enrich_dimensions(pd.concat([frame, pd.DataFrame(extra)], ignore_index=True))
    captured = []
    monkeypatch.setattr(app, "st", _FakeStreamlit())
    monkeypatch.setattr(app, "chart", lambda figure, key: captured.append((key, figure)))
    app.pareto_chart(frame)
    figure = dict(captured)["overview-pareto"]
    bars, cumulative = figure.data[0], figure.data[1]
    assert len(bars.x) == 11
    assert bars.x[-1] == "Others"
    assert abs(float(cumulative.y[-1]) - 1.0) < 1e-12
    assert float(bars.y[-1]) > 0
