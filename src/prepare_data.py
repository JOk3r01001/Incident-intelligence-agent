"""Prepare a new Olist database, or inspect an existing one without changing it.

This deterministic setup runs outside the AI tool loop. It replaces the old
exploratory importer and detector; analytical queries live in agent_tools.py.
"""
import argparse
from collections import defaultdict, deque
from datetime import date
import os
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "incident_intelligence.duckdb"
SOURCE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
}
REQUIRED_TABLES = {*SOURCE_FILES, "detected_anomalies"}
WINDOW_SIZE = 4
THRESHOLD = 30


def detect_anomalies(daily_sales: list[tuple[date, float]]) -> list[tuple]:
    """Apply the original median/30% detector to chronologically ordered sales."""
    history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
    anomalies = []
    for order_date, total_sales in daily_sales:
        actual = float(total_sales)
        previous = history[order_date.weekday()]
        # Compare with four prior observations of this weekday, not future data.
        # Missing sales days stay absent, matching the original detector.
        if len(previous) == WINDOW_SIZE:
            expected = median(previous)
            if expected > 0:
                deviation = (actual - expected) / expected * 100
                if abs(deviation) >= THRESHOLD:
                    anomalies.append((order_date, actual, expected, deviation))
        previous.append(actual)
    return anomalies


def inspect_database(db_path: Path) -> dict[str, int]:
    """Check required tables and count rows using a read-only connection."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        missing = REQUIRED_TABLES - {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if missing:
            raise RuntimeError(
                f"Existing database is missing tables: {', '.join(sorted(missing))}. "
                "It was not changed. Restore a prepared database, or build from CSVs "
                "using --db with a different, unused destination path."
            )
        # Identifiers come only from fixed application constants, never user text.
        return {table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in sorted(REQUIRED_TABLES)}


def prepare_database(data_dir: Path = DATA_DIR, db_path: Path = DB_PATH) -> dict:
    """Create a complete database only if the destination does not already exist."""
    data_dir, db_path = Path(data_dir), Path(db_path)
    # Treat even broken symlinks as occupied; no existing destination is replaced.
    if db_path.exists() or db_path.is_symlink():
        return {"created": False, "rows": inspect_database(db_path)}

    sources = {table: data_dir / filename for table, filename in SOURCE_FILES.items()}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Olist CSV files: " + ", ".join(missing)
            + ". Put the three source CSVs in data/ or pass --data-dir. "
            "You can also restore an existing prepared database."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Build beside the destination so publishing is on the same filesystem.
    # A failed import leaves no partial database at the requested destination.
    with TemporaryDirectory(prefix=".prepare-", dir=db_path.parent) as temporary:
        staging = Path(temporary) / "prepared.duckdb"
        with duckdb.connect(str(staging)) as con:
            for table, path in sources.items():
                # Only fixed table identifiers are interpolated; paths are parameters.
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto(?)", [str(path)])
            # Purchase date plus final delivered status matches the analytics tools.
            # Summing item prices excludes freight, canceled orders, and accounting revenue.
            daily_sales = con.execute("""
                SELECT CAST(o.order_purchase_timestamp AS DATE) AS order_date,
                       SUM(i.price) AS item_sales
                FROM orders o JOIN order_items i ON o.order_id = i.order_id
                WHERE o.order_status = 'delivered'
                GROUP BY order_date ORDER BY order_date
            """).fetchall()
            anomalies = detect_anomalies(daily_sales)
            con.execute("""CREATE TABLE detected_anomalies (
                order_date DATE, actual_sales DOUBLE,
                expected_sales DOUBLE, deviation_pct DOUBLE
            )""")
            # DuckDB rejects executemany with an empty parameter sequence.
            if anomalies:
                con.executemany("INSERT INTO detected_anomalies VALUES (?, ?, ?, ?)", anomalies)
        counts = inspect_database(staging)
        # Closing DuckDB flushes the WAL before publication. Unlike replace/rename,
        # link fails if another process creates the destination during our import.
        os.link(staging, db_path)
    return {"created": True, "rows": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing the three Olist CSVs")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Database destination; existing files are checked read-only")
    args = parser.parse_args()
    try:
        result = prepare_database(args.data_dir, args.db)
    except (OSError, RuntimeError, duckdb.Error) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")
    action = "Created" if result["created"] else "Checked existing database read-only; unchanged:"
    print(f"{action} {args.db}")
    for table, count in result["rows"].items():
        print(f"{table}: {count:,} rows")


if __name__ == "__main__":
    main()
