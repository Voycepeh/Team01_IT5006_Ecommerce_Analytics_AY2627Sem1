"""Build the Phase 1 dashboard dataset at one row per Olist order.

The workflow keeps every row from the raw orders table. One-to-many tables are
aggregated to ``order_id`` before they are joined, so item, payment, and review
records cannot multiply orders in the output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dashboard_orders.parquet"

ORDER_DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

EXPECTED_OUTPUT_COLUMNS = {
    "order_id",
    "purchase_timestamp",
    "purchase_date",
    "purchase_month",
    "customer_state",
    "seller_state",
    "product_category",
    "item_count",
    "total_item_value",
    "total_freight_value",
    "total_payment_value",
    "review_score",
    "delivery_days",
    "estimated_delivery_days",
    "days_late",
    "is_late",
    "is_negative_review",
}


def read_csv_checked(
    path: Path,
    required_columns: set[str],
    *,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Read a CSV and fail early when its expected schema is unavailable."""
    if not path.is_file():
        raise FileNotFoundError(f"Required raw file not found: {path}")

    frame = pd.read_csv(path, parse_dates=parse_dates)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return frame


def validate_unique(frame: pd.DataFrame, columns: list[str], table_name: str) -> None:
    """Validate a declared source or aggregate key."""
    if frame[columns].isna().any().any():
        raise ValueError(f"{table_name} has missing values in key {columns}")
    duplicate_count = int(frame.duplicated(columns).sum())
    if duplicate_count:
        raise ValueError(
            f"{table_name} has {duplicate_count:,} duplicate rows for key {columns}"
        )


def load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and validate the raw tables used by the dashboard workflow."""
    tables = {
        "orders": read_csv_checked(
            raw_dir / "olist_orders_dataset.csv",
            {
                "order_id",
                "customer_id",
                "order_status",
                *ORDER_DATE_COLUMNS,
            },
            parse_dates=ORDER_DATE_COLUMNS,
        ),
        "customers": read_csv_checked(
            raw_dir / "olist_customers_dataset.csv",
            {
                "customer_id",
                "customer_unique_id",
                "customer_city",
                "customer_state",
            },
        ),
        "items": read_csv_checked(
            raw_dir / "olist_order_items_dataset.csv",
            {
                "order_id",
                "order_item_id",
                "product_id",
                "seller_id",
                "price",
                "freight_value",
            },
        ),
        "payments": read_csv_checked(
            raw_dir / "olist_order_payments_dataset.csv",
            {
                "order_id",
                "payment_sequential",
                "payment_type",
                "payment_installments",
                "payment_value",
            },
        ),
        "reviews": read_csv_checked(
            raw_dir / "olist_order_reviews_dataset.csv",
            {
                "review_id",
                "order_id",
                "review_score",
                "review_creation_date",
                "review_answer_timestamp",
            },
            parse_dates=["review_creation_date", "review_answer_timestamp"],
        ),
        "products": read_csv_checked(
            raw_dir / "olist_products_dataset.csv",
            {"product_id", "product_category_name"},
        ),
        "sellers": read_csv_checked(
            raw_dir / "olist_sellers_dataset.csv",
            {"seller_id", "seller_state"},
        ),
        "translations": read_csv_checked(
            raw_dir / "product_category_name_translation.csv",
            {"product_category_name", "product_category_name_english"},
        ),
    }

    validate_unique(tables["orders"], ["order_id"], "orders")
    validate_unique(tables["customers"], ["customer_id"], "customers")
    validate_unique(tables["items"], ["order_id", "order_item_id"], "items")
    validate_unique(
        tables["payments"], ["order_id", "payment_sequential"], "payments"
    )
    validate_unique(tables["reviews"], ["review_id", "order_id"], "reviews")
    validate_unique(tables["products"], ["product_id"], "products")
    validate_unique(tables["sellers"], ["seller_id"], "sellers")
    validate_unique(
        tables["translations"], ["product_category_name"], "translations"
    )

    order_ids = set(tables["orders"]["order_id"])
    for table_name in ("items", "payments", "reviews"):
        orphan_count = int((~tables[table_name]["order_id"].isin(order_ids)).sum())
        if orphan_count:
            raise ValueError(
                f"{table_name} contains {orphan_count:,} records without a raw order"
            )

    return tables


def aggregate_dimension(
    frame: pd.DataFrame,
    value_column: str,
    primary_name: str,
    all_values_name: str,
    count_name: str,
    *,
    weight_column: str | None = None,
) -> pd.DataFrame:
    """Summarise a many-valued dimension using a deterministic dominant value.

    The primary value has the largest record count, or largest summed weight when
    ``weight_column`` is supplied. Ties are resolved alphabetically. The sorted
    pipe-delimited field retains every distinct value represented by the order.
    """
    columns = ["order_id", value_column]
    if weight_column:
        columns.append(weight_column)
    values = frame[columns].dropna(subset=[value_column]).copy()

    if values.empty:
        return pd.DataFrame(
            columns=["order_id", primary_name, all_values_name, count_name]
        )

    if weight_column:
        ranked = (
            values.groupby(["order_id", value_column], as_index=False)[weight_column]
            .sum()
            .rename(columns={weight_column: "_rank"})
        )
    else:
        ranked = (
            values.groupby(["order_id", value_column], as_index=False)
            .size()
            .rename(columns={"size": "_rank"})
        )

    ranked = ranked.sort_values(
        ["order_id", "_rank", value_column], ascending=[True, False, True]
    )
    primary = (
        ranked.drop_duplicates("order_id")[["order_id", value_column]]
        .rename(columns={value_column: primary_name})
    )
    all_values = (
        values[["order_id", value_column]]
        .drop_duplicates()
        .sort_values(["order_id", value_column])
        .groupby("order_id")[value_column]
        .agg(" | ".join)
        .rename(all_values_name)
        .reset_index()
    )
    counts = (
        values.groupby("order_id")[value_column]
        .nunique()
        .rename(count_name)
        .reset_index()
    )
    result = primary.merge(all_values, on="order_id", validate="1:1").merge(
        counts, on="order_id", validate="1:1"
    )
    validate_unique(result, ["order_id"], f"{value_column} summary")
    return result


def aggregate_items(
    items: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    translations: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate items, translated categories, and seller geography by order."""
    products_with_category = products.merge(
        translations,
        on="product_category_name",
        how="left",
        validate="m:1",
    )
    # Keep untranslated source labels visible rather than inventing translations.
    products_with_category["product_category"] = products_with_category[
        "product_category_name_english"
    ].fillna(products_with_category["product_category_name"])

    enriched = (
        items.merge(
            products_with_category[["product_id", "product_category"]],
            on="product_id",
            how="left",
            validate="m:1",
        )
        .merge(
            sellers[["seller_id", "seller_state"]],
            on="seller_id",
            how="left",
            validate="m:1",
        )
    )
    if len(enriched) != len(items):
        raise AssertionError("Product or seller enrichment changed the item row count")

    totals = (
        enriched.groupby("order_id", as_index=False)
        .agg(
            item_count=("order_item_id", "count"),
            total_item_value=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
            seller_count=("seller_id", "nunique"),
        )
    )
    category_summary = aggregate_dimension(
        enriched,
        "product_category",
        "product_category",
        "product_categories",
        "product_category_count",
    )
    seller_summary = aggregate_dimension(
        enriched,
        "seller_state",
        "seller_state",
        "seller_states",
        "seller_state_count",
    )
    result = totals.merge(category_summary, on="order_id", how="left", validate="1:1")
    result = result.merge(seller_summary, on="order_id", how="left", validate="1:1")
    validate_unique(result, ["order_id"], "item aggregates")
    return result


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate payment totals and payment-method information by order."""
    totals = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_count=("payment_sequential", "count"),
            total_payment_value=("payment_value", "sum"),
            max_payment_installments=("payment_installments", "max"),
        )
    )
    method_summary = aggregate_dimension(
        payments,
        "payment_type",
        "payment_method",
        "payment_methods",
        "payment_method_count",
        weight_column="payment_value",
    )
    result = totals.merge(method_summary, on="order_id", how="left", validate="1:1")
    validate_unique(result, ["order_id"], "payment aggregates")
    return result


def select_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Select one stable review per order, preferring the latest answered review."""
    review_counts = reviews.groupby("order_id").size().rename("review_count")
    selected = (
        reviews.sort_values(
            ["review_answer_timestamp", "review_creation_date", "review_id"],
            na_position="first",
        )
        .drop_duplicates("order_id", keep="last")
        [[
            "order_id",
            "review_id",
            "review_score",
            "review_creation_date",
            "review_answer_timestamp",
        ]]
        .merge(review_counts, on="order_id", validate="1:1")
    )
    validate_unique(selected, ["order_id"], "selected reviews")
    return selected


def add_derived_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Add dashboard dates, durations, and nullable outcome flags."""
    result = frame.rename(
        columns={"order_purchase_timestamp": "purchase_timestamp"}
    ).copy()
    result["purchase_date"] = result["purchase_timestamp"].dt.normalize()
    result["purchase_month"] = result["purchase_timestamp"].dt.to_period("M").astype(
        "string"
    )

    delivered_complete = (
        result["order_status"].eq("delivered")
        & result["order_delivered_customer_date"].notna()
    )
    result["is_delivered_complete"] = delivered_complete
    lateness_eligible = (
        delivered_complete & result["order_estimated_delivery_date"].notna()
    )

    delivery_days = (
        result["order_delivered_customer_date"] - result["purchase_timestamp"]
    ).dt.days
    days_late = (
        result["order_delivered_customer_date"]
        - result["order_estimated_delivery_date"]
    ).dt.days
    result["delivery_days"] = delivery_days.where(delivered_complete).astype("Float64")
    result["estimated_delivery_days"] = (
        result["order_estimated_delivery_date"] - result["purchase_timestamp"]
    ).dt.days.astype("Float64")
    result["days_late"] = days_late.where(lateness_eligible).astype("Float64")

    result["is_late"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result.loc[lateness_eligible, "is_late"] = (
        result.loc[lateness_eligible, "order_delivered_customer_date"]
        > result.loc[lateness_eligible, "order_estimated_delivery_date"]
    )
    result["is_negative_review"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    reviewed = result["review_score"].notna()
    result.loc[reviewed, "is_negative_review"] = result.loc[
        reviewed, "review_score"
    ].le(2)
    return result


def build_dashboard_orders(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create the validated one-row-per-order analytical table."""
    orders = tables["orders"]
    item_aggregates = aggregate_items(
        tables["items"],
        tables["products"],
        tables["sellers"],
        tables["translations"],
    )
    payment_aggregates = aggregate_payments(tables["payments"])
    selected_reviews = select_reviews(tables["reviews"])

    result = (
        orders.merge(
            tables["customers"][[
                "customer_id",
                "customer_unique_id",
                "customer_city",
                "customer_state",
            ]],
            on="customer_id",
            how="left",
            validate="m:1",
        )
        .merge(item_aggregates, on="order_id", how="left", validate="1:1")
        .merge(payment_aggregates, on="order_id", how="left", validate="1:1")
        .merge(selected_reviews, on="order_id", how="left", validate="1:1")
    )
    if len(result) != len(orders):
        raise AssertionError(
            f"Joining changed the order row count: {len(orders):,} -> {len(result):,}"
        )
    return add_derived_fields(result)


def validate_output(frame: pd.DataFrame, raw_orders: pd.DataFrame) -> int:
    """Run final grain, schema, type, and duration sanity checks."""
    duplicate_order_count = int(frame.duplicated("order_id").sum())
    if duplicate_order_count:
        raise AssertionError(
            f"Final output has {duplicate_order_count:,} duplicate order IDs"
        )
    if len(frame) > raw_orders["order_id"].nunique():
        raise AssertionError("Final row count exceeds the number of unique raw orders")

    missing_columns = EXPECTED_OUTPUT_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise AssertionError(f"Final output is missing columns: {sorted(missing_columns)}")
    for column in ("delivery_days", "estimated_delivery_days", "days_late"):
        if not is_numeric_dtype(frame[column]):
            raise TypeError(f"{column} must be numeric, got {frame[column].dtype}")

    complete = frame["is_delivered_complete"]
    if frame.loc[complete, "delivery_days"].isna().any():
        raise AssertionError("A complete delivered order is missing delivery_days")
    if frame.loc[~complete, "delivery_days"].notna().any():
        raise AssertionError("Undelivered or inconsistent orders have delivery_days")

    lateness_eligible = complete & frame["order_estimated_delivery_date"].notna()
    if frame.loc[lateness_eligible, ["days_late", "is_late"]].isna().any().any():
        raise AssertionError("An eligible delivered order is missing a lateness outcome")
    if frame.loc[~lateness_eligible, ["days_late", "is_late"]].notna().any().any():
        raise AssertionError("An order without complete delivery dates has lateness outcomes")
    if (frame.loc[complete, "delivery_days"] < 0).any():
        raise AssertionError("A complete order has a delivery timestamp before purchase")
    if (frame["estimated_delivery_days"].dropna() < 0).any():
        raise AssertionError("An estimated delivery date occurs before purchase")
    if not frame["review_score"].dropna().between(1, 5).all():
        raise AssertionError("Review scores outside the expected 1-5 range were found")
    return duplicate_order_count


def prepare_dashboard_data(raw_dir: Path, output_path: Path) -> pd.DataFrame:
    """Load raw data, build and validate the dataset, then write Parquet."""
    tables = load_raw_tables(raw_dir)
    dashboard_orders = build_dashboard_orders(tables)
    duplicate_count = validate_output(dashboard_orders, tables["orders"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_orders.to_parquet(output_path, index=False)

    print("Dashboard data preparation complete")
    print(f"Raw order count: {len(tables['orders']):,}")
    print(f"Final row count: {len(dashboard_orders):,}")
    print(f"Final column count: {dashboard_orders.shape[1]:,}")
    print(f"Duplicate order count: {duplicate_count:,}")
    print(f"Output path: {output_path.resolve()}")
    return dashboard_orders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the one-row-per-order Olist dashboard dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing the raw Olist CSV files (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Parquet output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dashboard_data(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
