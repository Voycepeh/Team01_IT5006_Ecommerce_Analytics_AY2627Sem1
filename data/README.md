# Data

The lecturer-provided, unchanged Olist source CSV files belong in:

```text
data/raw/
```

The Phase 1 dashboard preparation workflow writes:

```text
data/processed/dashboard_orders.parquet
```

Phase 1 EDA also provides `order_level.csv`, `item_level.csv`, `products_clean.csv`,
and `reviews_dedup.csv` in `data/processed/`. Phase 2 classification consumes the first
two CSVs as its cleaned analytical source layer to model negative-review risk among
late-delivered orders. It does not use the dashboard Parquet;
see [`docs/phase2-late-order-classification-workflow.md`](../docs/phase2-late-order-classification-workflow.md).

One row in this processed dataset represents one order. Item, payment, and review tables
are reduced to order level before joining so orders are not duplicated. For orders with
multiple product categories or seller states, the primary value is the most frequent item
value (alphabetical tie-break), while pipe-delimited summary columns retain all values.
The primary payment method is the method with the highest total payment value. If multiple
reviews exist, the most recently answered review is selected, with creation timestamp and
review ID providing deterministic tie-breaks.

Cancelled, undelivered, and internally inconsistent orders remain in the dataset. Actual
delivery measures and the late flag are null unless an order has status `delivered` and a
customer delivery timestamp. Missing reviews also retain a null negative-review flag.

Regenerate the processed dataset from the repository root with:

```bash
python src/prepare_dashboard_data.py
```

README files inside `data/` may still be committed so teammates know what each folder is for.
