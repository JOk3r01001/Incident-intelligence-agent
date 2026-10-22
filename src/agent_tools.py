"""Read-only analytical tools exposed to the local investigation agent."""
from datetime import date as Date
from pathlib import Path
from statistics import mean, median
from typing import Literal

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
    # bool is an int subclass; reject it along with coerced strings/floats.
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}.")
    return value


def _connect():
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"DuckDB database not found: {DB_PATH}. "
            "Run 'python src/prepare_data.py' from the project root first."
        )
    # Enforce read-only access at the database boundary, beyond the tool allowlist.
    con = duckdb.connect(str(DB_PATH), read_only=True)
    required = {"orders", "order_items", "products", "detected_anomalies"}
    missing = required - {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if missing:
        con.close()
        raise RuntimeError(
            f"DuckDB is missing tables: {', '.join(sorted(missing))}. "
            "Restore a prepared Olist database, or use 'python src/prepare_data.py "
            "--db data/rebuilt.duckdb' to build a separate database from CSVs. "
            "Existing databases are never overwritten by the setup command."
        )
    return con


def get_anomaly(day: str) -> dict:
    """Look up a detected sales anomaly for a specific YYYY-MM-DD date.

    Args:
        day: The date to investigate in YYYY-MM-DD format.
    """
    day = _validated_date(day)
    with _connect() as con:
        # Bind values separately from SQL; model text never becomes executable SQL.
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
    # The join repeats orders with multiple items: count distinct order IDs.
    # Group by purchase date and final recorded status, not delivery date.
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
    # Strictly earlier dates prevent the selected anomaly contaminating its baseline.
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
    # A left join retains sales whose product/category metadata is missing.
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


def get_dataset_summary() -> dict:
    """Get historical date coverage, order count, and stored anomaly count."""
    with _connect() as con:
        first, last, orders = con.execute("""SELECT MIN(CAST(order_purchase_timestamp AS DATE)),
            MAX(CAST(order_purchase_timestamp AS DATE)), COUNT(*) FROM orders""").fetchone()
        anomalies = con.execute("SELECT COUNT(*) FROM detected_anomalies").fetchone()[0]
    return {"first_order_date": first.isoformat() if first else None,
            "last_order_date": last.isoformat() if last else None,
            "orders": int(orders), "stored_anomalies": int(anomalies),
            "note": "Historical data. Anomalies are stored statistical flags, not confirmed incidents."}


def find_anomalies(start_date: str | None = None, end_date: str | None = None,
                   direction: Literal["both", "spikes", "drops"] = "both",
                   sort_by: Literal["deviation", "sales", "date"] = "deviation",
                   limit: int = 5, offset: int = 0) -> dict:
    """Find stored sales anomalies with optional date filters and bounded pagination.

    Args:
        start_date: Inclusive first date in YYYY-MM-DD format, or omit for all history.
        end_date: Inclusive last date in YYYY-MM-DD format, or omit for all history.
        direction: both, spikes (positive deviation), or drops (negative deviation).
        sort_by: deviation (absolute percentage), sales (actual item sales), or date (newest first).
        limit: Number of anomalies to return, from 1 to 10.
        offset: Number of matching anomalies to skip, from 0 to 10000.
    """
    start_date = _validated_date(start_date) if start_date is not None else None
    end_date = _validated_date(end_date) if end_date is not None else None
    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")
    if direction not in ("both", "spikes", "drops"):
        raise ValueError("direction must be 'both', 'spikes', or 'drops'.")
    ordering = {"deviation": "ABS(deviation_pct) DESC, order_date DESC",
                "sales": "actual_sales DESC, order_date DESC", "date": "order_date DESC"}
    if not isinstance(sort_by, str) or sort_by not in ordering:
        raise ValueError("sort_by must be 'deviation', 'sales', or 'date'.")
    limit = _bounded_integer(limit, "limit", 10)
    if type(offset) is not int or not 0 <= offset <= 10000:
        raise ValueError("offset must be an integer between 0 and 10000.")
    # All user/model values are bound parameters. Only an allowlisted ORDER BY
    # fragment is interpolated; the model cannot supply SQL or rewrite the detector.
    where = """WHERE (CAST(? AS DATE) IS NULL OR order_date >= CAST(? AS DATE))
        AND (CAST(? AS DATE) IS NULL OR order_date <= CAST(? AS DATE))
        AND (? = 'both' OR (? = 'spikes' AND deviation_pct > 0)
             OR (? = 'drops' AND deviation_pct < 0))"""
    params = [start_date, start_date, end_date, end_date, direction, direction, direction]
    with _connect() as con:
        total = con.execute("SELECT COUNT(*) FROM detected_anomalies " + where, params).fetchone()[0]
        rows = con.execute("SELECT order_date, actual_sales, expected_sales, deviation_pct "
                           "FROM detected_anomalies " + where + f" ORDER BY {ordering[sort_by]} LIMIT ? OFFSET ?",
                           params + [limit, offset]).fetchall()
    return {"anomalies": [{"date": row[0].isoformat(), "actual_sales": round(float(row[1]), 2),
                           "expected_sales": round(float(row[2]), 2), "deviation_pct": round(float(row[3]), 2)}
                          for row in rows],
            "total_matches": int(total), "limit": limit, "offset": offset,
            "has_more": offset + len(rows) < total, "sort_by": sort_by, "direction": direction,
            "start_date": start_date, "end_date": end_date,
            "note": "Stored detector flags; percentage deviation is not business impact or proof of an incident."}


def create_graph(graph_type: Literal["daily_sales", "daily_orders", "daily_average_order_value",
                                      "category_sales", "order_status"],
                 start_date: str, end_date: str | None = None, limit: int = 10) -> dict:
    """Create a database-backed line/bar graph for display in Streamlit when requested.

    Args:
        graph_type: daily_sales, daily_orders, daily_average_order_value, category_sales, or order_status.
        start_date: Inclusive first purchase date in YYYY-MM-DD format.
        end_date: Inclusive last purchase date, at most 90 days including start; omit for a single day.
        limit: Maximum categories or statuses to display, from 1 to 15; default 10.
    """
    start_date = _validated_date(start_date)
    end_date = _validated_date(end_date) if end_date is not None else start_date
    days = (Date.fromisoformat(end_date) - Date.fromisoformat(start_date)).days + 1
    if not 1 <= days <= 90:
        raise ValueError("Graph dates must be ordered and cover at most 90 days inclusive. Request a smaller range.")
    limit = _bounded_integer(limit, "limit", 15)
    metrics = {"daily_sales": ("item_sales", "Daily delivered-order item sales", "Item sales (R$)"),
               "daily_orders": ("orders", "Daily distinct delivered orders", "Delivered orders"),
               "daily_average_order_value": ("average_order_value", "Daily average delivered-order value", "Average order value (R$)")}
    if not isinstance(graph_type, str) or graph_type not in (*metrics, "category_sales", "order_status"):
        raise ValueError("graph_type must be daily_sales, daily_orders, daily_average_order_value, category_sales, or order_status.")

    truncated = False
    # The model supplies only validated values/enums. All SQL is fixed and
    # parameterized; the returned chart description is generated by Python.
    with _connect() as con:
        if graph_type in metrics:
            rows = con.execute("""SELECT CAST(o.order_purchase_timestamp AS DATE) AS day,
                    SUM(i.price), COUNT(DISTINCT o.order_id)
                FROM orders o JOIN order_items i ON o.order_id = i.order_id
                WHERE o.order_status = 'delivered'
                  AND CAST(o.order_purchase_timestamp AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                GROUP BY day ORDER BY day""", [start_date, end_date]).fetchall()
            y, title, y_label = metrics[graph_type]
            data = []
            for day, sales, orders in rows:
                if graph_type == "daily_orders":
                    value = orders
                elif sales is None:
                    value = None
                elif graph_type == "daily_average_order_value":
                    value = float(sales) / orders
                else:
                    value = float(sales)
                data.append({"date": day.isoformat(), y: round(value, 2) if value is not None else None})
            x = "date"
            # A single observation is visible as a bar, rather than an empty line.
            chart_type = "line" if len(data) > 1 else "bar"
            note = ("Final delivered orders grouped by purchase date. Item sales exclude freight. "
                    "Days with no matching items are omitted, not filled with invented zeros. "
                    "Missing prices remain unknown. These values do not establish causes.")
        elif graph_type == "category_sales":
            rows = con.execute("""SELECT COALESCE(p.product_category_name, 'unknown') AS category, SUM(i.price) AS sales
                FROM orders o JOIN order_items i ON o.order_id = i.order_id
                LEFT JOIN products p ON i.product_id = p.product_id
                WHERE o.order_status = 'delivered'
                  AND CAST(o.order_purchase_timestamp AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                GROUP BY category ORDER BY sales DESC NULLS LAST, category LIMIT ?""",
                               [start_date, end_date, limit + 1]).fetchall()
            truncated = len(rows) > limit
            data = [{"category": row[0], "item_sales": round(float(row[1]), 2) if row[1] is not None else None}
                    for row in rows[:limit]]
            x, y, chart_type = "category", "item_sales", "bar"
            title, y_label = "Top categories by delivered-order item sales", "Item sales (R$)"
            note = ("Delivered-order item prices grouped by purchase date; freight excluded. "
                    "Only the leading categories are shown. Omitted categories are not zero, "
                    "and a ranking alone does not establish abnormal category demand.")
        else:
            rows = con.execute("""SELECT order_status, COUNT(*) AS orders FROM orders
                WHERE CAST(order_purchase_timestamp AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                GROUP BY order_status ORDER BY orders DESC, order_status LIMIT ?""",
                               [start_date, end_date, limit + 1]).fetchall()
            truncated = len(rows) > limit
            data = [{"status": row[0] or "unknown", "orders": int(row[1])} for row in rows[:limit]]
            x, y, chart_type = "status", "orders", "bar"
            title, y_label = "Orders by final recorded status", "Orders"
            note = ("All orders grouped by final recorded status and purchase date, including canceled orders. "
                    "These are not statuses observed on that historical day and cannot establish delays or conversion rates.")
    if not data:
        note = "No matching database observations for this graph and date range."
    elif not any(row[y] is not None for row in data):
        data = []
        note = "Matching orders have no known prices for this metric; no values were invented and no graph is plotted."
    known_values = [row[y] for row in data if row[y] is not None]
    maximum = max(known_values) if known_values else None
    summary = {"observations": len(data), "minimum": min(known_values) if known_values else None,
               "maximum": maximum,
               "maximum_at": [row[x] for row in data if row[y] == maximum][:3] if known_values else []}
    return {"graph_type": graph_type, "chart_type": chart_type,
            "title": f"{title}: {start_date} to {end_date}", "x": x, "y": y, "y_label": y_label,
            "start_date": start_date, "end_date": end_date, "data": data, "truncated": truncated,
            "summary": summary, "note": note}


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
