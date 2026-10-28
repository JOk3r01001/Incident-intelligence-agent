"""Create a clearly synthetic warehouse dataset without replacing any existing file."""
import argparse
from datetime import date, timedelta
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def prepare_demo(destination: Path):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Destination already exists: {destination}. No file was changed.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    start = date(2025, 1, 1)
    for offset in range(59):
        day = start + timedelta(days=offset)
        for warehouse, base in (("North", 100), ("South", 60)):
            units = base * (3 if day == date(2025, 2, 14) else 1)
            # Different per-unit handling times make weighted ratios testable.
            minutes = units * (2 if warehouse == "North" else 3)
            rows.append((f"{day}-{warehouse}", day, warehouse, units, minutes, True))
    with TemporaryDirectory(prefix=".warehouse-demo-", dir=destination.parent) as directory:
        temporary = Path(directory) / "demo.duckdb"
        with duckdb.connect(str(temporary)) as con:
            con.execute("CREATE TABLE warehouse_shifts(shift_id VARCHAR, work_date DATE, warehouse VARCHAR, "
                        "shipped_units INTEGER, handling_minutes INTEGER, completed BOOLEAN)")
            con.executemany("INSERT INTO warehouse_shifts VALUES (?, ?, ?, ?, ?, ?)", rows)
        os.link(temporary, destination)  # Fails safely if the destination appeared concurrently.
    return {"database": str(destination), "rows": len(rows), "synthetic": True,
            "reference": {"date": "2025-02-14", "units_shipped": 480, "completed_shifts": 2,
                          "handling_time_per_unit": 2.375, "previous_friday_units": 160}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "warehouse_demo.duckdb")
    args = parser.parse_args()
    try:
        print(prepare_demo(args.db))
    except (OSError, duckdb.Error) as exc:
        parser.exit(1, f"Demo preparation failed: {exc}\n")
