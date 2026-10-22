"""Read-only analytical tools exposed to the local investigation agent."""
from datetime import date as Date
from pathlib import Path
from statistics import mean, median

import duckdb

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "incident_intelligence.duckdb"


def _validated_date(day: str) -> str:
    """Require an ISO-formatted date; prevents ambiguous date handling."""
    if not isinstance(day, str):
        raise ValueError("day must be a date string in YYYY-MM-DD format.")
    try:
        parsed = Date.fromisoformat(day).isoformat()
    except ValueError as exc:
        raise ValueError("day must be a valid date in YYYY-MM-DD format.") from exc
    if parsed != day:
        raise ValueError("day must use YYYY-MM-DD format.")
    return parsed


def _bounded_integer(value: int, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}.")
    return value


def _connect():
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"DuckDB database not found: {DB_PATH}. "
            "Run 'python src/explore_data.py' from the project root first."
        )
    con = duckdb.connect(str(DB_PATH), read_only=True)
    required = {"orders", "order_items", "products", "detected_anomalies"}
    missing = required - {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if missing:
        con.close()
        raise RuntimeError(
            f"DuckDB is missing tables: {', '.join(sorted(missing))}. "
            "Restore the prepared Olist database; if only detected_anomalies is missing, "
            "run 'python src/detect_anomalies.py' from the repository root."
        )
    return con


def get_anomaly(day: str) -> dict:
    """Look up a detected sales anomaly for a specific YYYY-MM-DD date.

    Args:
        day: The date to investigate in YYYY-MM-DD format.
    """
    day = _validated_date(day)
    with _connect() as con:
        row = con.execute(
            """SELECT actual_sales, expected_sales, deviation_pct
               FROM detected_anomalies WHERE order_date = ?""",
            [day],
        ).fetchone()
    if row is None:
        return {"date": day, "found": False}
    return {
        "date": day, "found": True,
        "actual_sales": round(float(row[0]), 2),
        "expected_sales": round(float(row[1]), 2),
        "deviation_pct": round(float(row[2]), 2),
    }


def get_daily_metrics(day: str) -> dict:
    """Get delivered-order item sales, unique orders and average order value for a date.

    Args:
        day: The date to analyze in YYYY-MM-DD format.
    """
    day = _validated_date(day)
    with _connect() as con:
        row = con.execute(
            """SELECT SUM(i.price), COUNT(DISTINCT o.order_id), COUNT(*)
               FROM orders AS o
               JOIN order_items AS i ON o.order_id = i.order_id
               WHERE o.order_status = 'delivered'
                 AND CAST(o.order_purchase_timestamp AS DATE) = ?""",
            [day],
        ).fetchone()
    revenue = round(float(row[0]), 2) if row[0] is not None else 0.0
    orders = int(row[1])
    return {
        "date": day, "item_sales": revenue, "orders": orders,
        "items": int(row[2]),
        "average_order_value": round(revenue / orders, 2) if orders else None,
    }


def get_weekday_history(day: str, weeks: int = 4) -> dict:
    """Get prior matching weekdays and compute historical sales/order baselines.

    Args:
        day: Target date in YYYY-MM-DD format.
        weeks: Number of previous matching weekdays to consider (1 to 8).
    """
    day = _validated_date(day)
    weeks = _bounded_integer(weeks, "weeks", 8)
    with _connect() as con:
        rows = con.execute(
            """SELECT CAST(o.order_purchase_timestamp AS DATE) AS order_date,
                      SUM(i.price) AS item_sales,
                      COUNT(DISTINCT o.order_id) AS orders
               FROM orders AS o
               JOIN order_items AS i ON o.order_id = i.order_id
               WHERE o.order_status = 'delivered'
                 AND CAST(o.order_purchase_timestamp AS DATE) < CAST(? AS DATE)
                 AND DAYNAME(o.order_purchase_timestamp) = DAYNAME(CAST(? AS DATE))
               GROUP BY order_date
               ORDER BY order_date DESC
               LIMIT ?""",
            [day, day, weeks],
        ).fetchall()
    observations = [
        {"date": r[0].isoformat(), "item_sales": round(float(r[1]), 2), "orders": int(r[2])}
        for r in rows
    ]
    sales = [r["item_sales"] for r in observations]
    orders = [r["orders"] for r in observations]
    return {
        "target_date": day,
        "sample_size": len(observations),
        "previous_matching_weekdays": observations,
        "mean_item_sales": round(mean(sales), 2) if sales else None,
        "median_item_sales": round(median(sales), 2) if sales else None,
        "mean_orders": round(mean(orders), 2) if orders else None,
        "caution": "Historical baselines are unreliable with few samples or rapidly changing trends.",
    }


def get_sales_by_category(day: str, limit: int = 10) -> dict:
    """List the categories with highest item sales for a given purchase date.

    Args:
        day: Date in YYYY-MM-DD format.
        limit: Maximum number of categories to return (1 to 15).
    """
    day = _validated_date(day)
    limit = _bounded_integer(limit, "limit", 15)
    with _connect() as con:
        rows = con.execute(
            """SELECT COALESCE(p.product_category_name, 'unknown') AS category,
                      SUM(i.price) AS item_sales,
                      COUNT(DISTINCT o.order_id) AS orders,
                      COUNT(*) AS items
               FROM orders AS o
               JOIN order_items AS i ON o.order_id = i.order_id
               LEFT JOIN products AS p ON i.product_id = p.product_id
               WHERE o.order_status = 'delivered'
                 AND CAST(o.order_purchase_timestamp AS DATE) = ?
               GROUP BY category
               ORDER BY item_sales DESC
               LIMIT ?""",
            [day, limit],
        ).fetchall()
    return {
        "date": day,
        "categories": [
            {"category": r[0], "item_sales": round(float(r[1]), 2),
             "orders_in_category": int(r[2]), "items": int(r[3])}
            for r in rows
        ],
        "note": "Orders containing multiple categories can appear in multiple category counts.",
    }


def get_order_status_breakdown(day: str) -> dict:
    """Count orders by final recorded status for a given purchase date.

    Args:
        day: Date in YYYY-MM-DD format.
    """
    day = _validated_date(day)
    with _connect() as con:
        rows = con.execute(
            """SELECT order_status, COUNT(*) AS orders
               FROM orders
               WHERE CAST(order_purchase_timestamp AS DATE) = ?
               GROUP BY order_status ORDER BY orders DESC""",
            [day],
        ).fetchall()
    return {"date": day, "statuses": [
        {"status": r[0], "orders": int(r[1])} for r in rows
    ]}


def list_anomalies() -> list[dict]:
    """List stored anomaly dates and their deviations for the dashboard."""
    with _connect() as con:
        rows = con.execute(
            """SELECT order_date, actual_sales, expected_sales, deviation_pct
               FROM detected_anomalies ORDER BY ABS(deviation_pct) DESC"""
        ).fetchall()
    return [
        {"date": r[0].isoformat(), "actual_sales": round(float(r[1]), 2),
         "expected_sales": round(float(r[2]), 2),
         "deviation_pct": round(float(r[3]), 2)}
        for r in rows
    ]


def get_sales_window(day: str) -> list[dict]:
    """Get daily sales in the 14 days before and 7 days after a date for charting."""
    day = _validated_date(day)
    with _connect() as con:
        rows = con.execute(
            """SELECT CAST(o.order_purchase_timestamp AS DATE) AS order_date,
                      SUM(i.price) AS item_sales
               FROM orders o JOIN order_items i ON o.order_id = i.order_id
               WHERE o.order_status = 'delivered'
                 AND CAST(o.order_purchase_timestamp AS DATE)
                     BETWEEN CAST(? AS DATE) - INTERVAL 14 DAY
                         AND CAST(? AS DATE) + INTERVAL 7 DAY
               GROUP BY order_date ORDER BY order_date""",
            [day, day],
        ).fetchall()
    return [{"date": r[0].isoformat(), "item_sales": round(float(r[1]), 2)} for r in rows]
