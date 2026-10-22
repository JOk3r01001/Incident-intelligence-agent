"""Verify setup against disposable CSVs/databases, never the user's dataset."""
import csv
from datetime import date, timedelta
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent_tools
import prepare_data


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "csvs"
        self.data.mkdir()
        self.db = Path(self.tmp.name) / "output" / "fresh.duckdb"
        # Four ordinary Fridays precede a spike. The spike has two orders and
        # three items; a high-priced canceled order must contribute no sales.
        rows = [(f"prior-{n}", "delivered", f"2017-11-{day:02}")
                for n, day in enumerate((3, 10, 17))]
        rows.insert(0, ("prior-oct", "delivered", "2017-10-27"))
        rows.extend([("a", "delivered", "2017-11-24"),
                     ("b", "delivered", "2017-11-24"),
                     ("c", "canceled", "2017-11-24")])
        self.write_csv("orders", ["order_id", "order_status", "order_purchase_timestamp"], rows)
        self.write_csv("order_items", ["order_id", "product_id", "price"], [
            ("prior-oct", "p", 10), ("prior-0", "p", 20),
            ("prior-1", "p", 30), ("prior-2", "p", 40),
            ("a", "p", 60), ("a", "p", 40), ("b", "p", 50), ("c", "p", 900),
        ])
        self.write_csv("products", ["product_id", "product_category_name"], [("p", "books")])

    def write_csv(self, table, columns, rows):
        path = self.data / prepare_data.SOURCE_FILES[table]
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_fresh_database_supports_existing_analytics(self):
        result = prepare_data.prepare_database(self.data, self.db)
        self.assertTrue(result["created"])
        self.assertEqual(result["rows"], {"orders": 7, "order_items": 8,
                                          "products": 1, "detected_anomalies": 1})
        with patch.object(agent_tools, "DB_PATH", self.db):
            metrics = agent_tools.get_daily_metrics("2017-11-24")
            self.assertEqual((metrics["item_sales"], metrics["orders"], metrics["items"]), (150, 2, 3))
            anomaly = agent_tools.get_anomaly("2017-11-24")
            self.assertEqual(anomaly["expected_sales"], 25)
            self.assertEqual(anomaly["deviation_pct"], 500)
            self.assertEqual(agent_tools.get_weekday_history("2017-11-24")["median_item_sales"], 25)
            self.assertEqual(agent_tools.get_sales_by_category("2017-11-24")["categories"][0]["category"], "books")

    def test_existing_database_is_unchanged_even_without_csvs(self):
        prepare_data.prepare_database(self.data, self.db)
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        with patch.object(prepare_data.duckdb, "connect", wraps=duckdb.connect) as connect:
            result = prepare_data.prepare_database(self.data / "absent", self.db)
        self.assertFalse(result["created"])
        self.assertEqual(connect.call_args.kwargs, {"read_only": True})
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), before)

    def test_incomplete_existing_database_is_not_repaired_or_overwritten(self):
        self.db.parent.mkdir()
        with duckdb.connect(str(self.db)) as con:
            con.execute("CREATE TABLE original_data (value INTEGER)")
        before = self.db.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "different, unused destination"):
            prepare_data.prepare_database(self.data, self.db)
        self.assertEqual(self.db.read_bytes(), before)

    def test_missing_csv_leaves_no_database(self):
        (self.data / prepare_data.SOURCE_FILES["products"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "olist_products_dataset.csv"):
            prepare_data.prepare_database(self.data, self.db)
        self.assertFalse(self.db.exists())

    def test_failed_import_leaves_no_partial_database_or_staging_files(self):
        self.write_csv("orders", ["wrong_column"], [("invalid",)])
        with self.assertRaises(duckdb.Error):
            prepare_data.prepare_database(self.data, self.db)
        self.assertFalse(self.db.exists())
        self.assertEqual(list(self.db.parent.iterdir()), [])

    def test_publication_cannot_replace_a_concurrently_created_destination(self):
        original_link = prepare_data.os.link

        def concurrent_link(source, destination):
            destination.write_bytes(b"another process created this file")
            return original_link(source, destination)

        with patch.object(prepare_data.os, "link", side_effect=concurrent_link):
            with self.assertRaises(FileExistsError):
                prepare_data.prepare_database(self.data, self.db)
        self.assertEqual(self.db.read_bytes(), b"another process created this file")
        self.assertEqual(list(self.db.parent.iterdir()), [self.db])

    def test_broken_destination_symlink_is_not_replaced(self):
        self.db.parent.mkdir()
        absent_target = self.db.parent / "absent.duckdb"
        self.db.symlink_to(absent_target)
        with self.assertRaises(duckdb.Error):
            prepare_data.prepare_database(self.data, self.db)
        self.assertTrue(self.db.is_symlink())
        self.assertFalse(absent_target.exists())

    def test_no_delivered_orders_produces_valid_empty_anomaly_table(self):
        self.write_csv("orders", ["order_id", "order_status", "order_purchase_timestamp"],
                       [("c", "canceled", "2017-11-24")])
        result = prepare_data.prepare_database(self.data, self.db)
        self.assertEqual(result["rows"]["detected_anomalies"], 0)
        with patch.object(agent_tools, "DB_PATH", self.db):
            self.assertEqual(agent_tools.list_anomalies(), [])


class DetectorTests(unittest.TestCase):
    def fridays(self, values):
        first = date(2017, 10, 27)
        return [(first + timedelta(weeks=n), value) for n, value in enumerate(values)]

    def test_four_prior_weekday_observations_and_inclusive_threshold(self):
        rows = self.fridays([100, 100, 100, 100, 130, 70])
        anomalies = prepare_data.detect_anomalies(rows)
        self.assertEqual([(row[0], row[2], row[3]) for row in anomalies],
                         [(rows[4][0], 100, 30), (rows[5][0], 100, -30)])
        self.assertEqual(prepare_data.detect_anomalies(self.fridays([100] * 4 + [129, 71])), [])

    def test_baseline_uses_only_prior_values_of_the_same_weekday(self):
        rows = self.fridays([10, 20, 30, 40, 150])
        rows.extend((day + timedelta(days=1), 9999) for day, _ in rows.copy())
        anomalies = prepare_data.detect_anomalies(sorted(rows))
        self.assertEqual(anomalies, [(date(2017, 11, 24), 150, 25, 500)])

    def test_zero_baseline_and_insufficient_history_do_not_flag(self):
        self.assertEqual(prepare_data.detect_anomalies(self.fridays([0, 0, 0, 0, 100])), [])
        self.assertEqual(prepare_data.detect_anomalies(self.fridays([1, 1, 9999])), [])


if __name__ == "__main__":
    unittest.main()
