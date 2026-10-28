"""Reusable read-only analytics compiled from approved business definitions."""
from collections import defaultdict
import argparse
from contextlib import contextmanager
from datetime import date, timedelta
import json
import math
from statistics import median
from threading import Timer
from typing import Literal

import duckdb

from agent_tools import _bounded_integer, _validated_date
from dataset_config import DatasetProfile, Metric, Term, column_sql, quoted


class BusinessAnalytics:
    QUERY_TIMEOUT = 10.0

    def __init__(self, profile: DatasetProfile):
        self.profile = profile
        self.definition = profile.definition
        self._validate_database()

    @contextmanager
    def connection(self):
        # No writable connection, migrations, temp views, or model SQL are used.
        self.profile.check_unchanged()
        with duckdb.connect(str(self.profile.database), read_only=True) as con:
            yield con
        self.profile.check_unchanged()

    def _execute(self, con, sql: str, parameters=()):
        timer = Timer(self.QUERY_TIMEOUT, con.interrupt)
        timer.daemon = True
        timer.start()
        try:
            return con.execute(sql, list(parameters)).fetchall()
        except duckdb.InterruptException as exc:
            raise RuntimeError(f"Analytical query exceeded {self.QUERY_TIMEOUT:g}s. Narrow the request.") from exc
        finally:
            timer.cancel()
            timer.join()

    def _validate_database(self):
        columns = {}
        with self.connection() as con:
            for alias, table in self.definition.tables.items():
                exists = self._execute(con, "SELECT 1 FROM information_schema.tables WHERE table_schema='main' "
                                       "AND table_name=? AND table_type='BASE TABLE'", [table.table])
                if not exists:
                    raise ValueError(f"Configured table {table.table!r} is absent or is not a base table.")
                columns[alias] = dict(self._execute(con, "SELECT column_name, data_type FROM information_schema.columns "
                                                   "WHERE table_schema='main' AND table_name=?", [table.table]))
                if any(key not in columns[alias] for key in table.grain):
                    raise ValueError(f"Grain for {alias} references a missing column.")
                keys = ", ".join(quoted(key) for key in table.grain)
                nulls = " OR ".join(f"{quoted(key)} IS NULL" for key in table.grain)
                if self._execute(con, f"SELECT 1 FROM {quoted(table.table)} WHERE {nulls} LIMIT 1"):
                    raise ValueError(f"Configured grain for {alias} contains NULL keys.")
                if self._execute(con, f"SELECT 1 FROM {quoted(table.table)} GROUP BY {keys} HAVING COUNT(*) > 1 LIMIT 1"):
                    raise ValueError(f"Configured grain for {alias} is not unique; joins or counts could duplicate facts.")

        def data_type(reference):
            alias, column = reference.split(".")
            if alias not in columns or column not in columns[alias]:
                raise ValueError(f"Configured column {reference!r} does not exist.")
            return columns[alias][column]

        for join in self.definition.joins.values():
            if data_type(join.left) != data_type(join.right):
                raise ValueError(f"Join columns {join.left} and {join.right} must have matching types.")
        for name, metric in self.definition.metrics.items():
            kind = data_type(metric.date)
            if kind != "DATE" and not kind.startswith("TIMESTAMP"):
                raise ValueError(f"Metric {name} requires a DATE or TIMESTAMP column, not {kind}.")
            for reference in [*metric.dimensions.values(), *(f.column for f in metric.filters)]:
                data_type(reference)
            for term in metric.calculation.terms():
                if term.column:
                    kind = data_type(term.column)
                    if term.aggregation in ("sum", "average") and not any(
                        kind.startswith(prefix) for prefix in
                        ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
                         "UINTEGER", "UBIGINT", "UHUGEINT", "DECIMAL", "FLOAT", "DOUBLE")
                    ):
                        raise ValueError(f"Metric {name} cannot aggregate non-numeric column {term.column}.")

    def metric(self, name: str) -> Metric:
        if not isinstance(name, str) or name not in self.definition.metrics:
            raise ValueError(f"Unknown metric. Choose one of: {', '.join(self.definition.metrics)}.")
        return self.definition.metrics[name]

    def _from(self, metric: Metric) -> str:
        tables = self.definition.tables
        result = f"FROM {quoted(tables[metric.source].table)} AS {quoted(metric.source)}"
        for name in metric.joins:
            join = self.definition.joins[name]
            alias = join.right.split(".")[0]
            result += (f" {join.kind.upper()} JOIN {quoted(tables[alias].table)} AS {quoted(alias)} "
                       f"ON {column_sql(join.left)} = {column_sql(join.right)}")
        return result

    def _where(self, metric: Metric, start: str | None = None, end: str | None = None):
        conditions, values = [], []
        for item in metric.filters:
            if item.value is None:
                conditions.append(f"{column_sql(item.column)} IS NULL")
            else:
                conditions.append(f"{column_sql(item.column)} = ?")
                values.append(item.value)
        if start is not None:
            conditions.append(f"CAST({column_sql(metric.date)} AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)")
            values.extend([start, end])
        return ("WHERE " + " AND ".join(conditions) if conditions else ""), values

    @staticmethod
    def _term_sql(term: Term) -> str:
        if term.aggregation == "count":
            return "COUNT(*)"
        column = column_sql(term.column)
        return {"sum": f"SUM({column})", "average": f"AVG({column})",
                "count_distinct": f"COUNT(DISTINCT {column})"}[term.aggregation]

    def _calculation(self, metric: Metric):
        terms = metric.calculation.terms()
        expression = self._term_sql(terms[0])
        if metric.calculation.aggregation == "ratio":
            expression = f"CAST({expression} AS DOUBLE) / NULLIF({self._term_sql(terms[1])}, 0)"
        # Missing values are reported alongside partial aggregates, never filled.
        refs = sorted({term.column for term in terms if term.column})
        nulls = " OR ".join(f"{column_sql(ref)} IS NULL" for ref in refs)
        missing = f"COUNT(*) FILTER (WHERE {nulls})" if nulls else "0"
        return expression, missing

    @staticmethod
    def date_range(start: str, end: str, maximum: int = 366):
        start, end = _validated_date(start), _validated_date(end)
        days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
        if not 1 <= days <= maximum:
            raise ValueError(f"Dates must be ordered and cover at most {maximum} days inclusive.")
        return start, end, days

    @staticmethod
    def _number(value):
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Metric aggregation contains non-finite values. Correct the source data before analysis.")
        return number

    def _query(self, name, start, end, grouping=None, limit=10, *, round_values=True, daily=False):
        metric = self.metric(name)
        expression, missing = self._calculation(metric)
        where, parameters = self._where(metric, start, end)
        prefix = ""
        suffix = ""
        if daily:
            prefix = f"CAST({column_sql(metric.date)} AS DATE) AS bucket, "
            suffix = " GROUP BY bucket ORDER BY bucket"
        elif grouping:
            if grouping not in metric.dimensions:
                raise ValueError(f"Unknown dimension for {name}. Choose one of: {', '.join(metric.dimensions) or 'none'}.")
            prefix = f"CAST({column_sql(metric.dimensions[grouping])} AS VARCHAR) AS bucket, "
            suffix = " GROUP BY bucket ORDER BY value DESC NULLS LAST, bucket LIMIT ?"
            parameters.append(limit + 1)
        sql = f"SELECT {prefix}{expression} AS value, COUNT(*) AS records, {missing} AS missing_values {self._from(metric)} {where}{suffix}"
        with self.connection() as con:
            rows = self._execute(con, sql, parameters)
        result = []
        for row in rows:
            bucket, value, records, absent = (row if daily or grouping else (None, *row))
            value = self._number(value)
            if not records:
                value = None  # No observations is not a measured zero, even for COUNT.
            item = {"value": round(value, metric.precision) if value is not None and round_values else value,
                    "records": int(records), "missing_values": int(absent)}
            if daily or grouping:
                item["date" if daily else "dimension"] = bucket.isoformat() if daily else bucket
            result.append(item)
        truncated = bool(grouping and not daily and len(result) > limit)
        if truncated:
            result = result[:limit]
        return result, truncated, {"sql": sql, "parameters": parameters, "read_only": True}

    def describe(self):
        metrics = {}
        with self.connection() as con:
            for name, metric in self.definition.metrics.items():
                where, params = self._where(metric)
                field = f"CAST({column_sql(metric.date)} AS DATE)"
                first, last, records, missing_dates = self._execute(con,
                    f"SELECT MIN({field}), MAX({field}), COUNT(*), COUNT(*) FILTER (WHERE {field} IS NULL) "
                    f"{self._from(metric)} {where}", params)[0]
                metrics[name] = {"label": metric.label, "description": metric.description, "unit": metric.unit,
                                 "aggregation": metric.calculation.aggregation, "dimensions": list(metric.dimensions),
                                 "first_date": first.isoformat() if first else None,
                                 "last_date": last.isoformat() if last else None,
                                 "records": int(records), "missing_dates": int(missing_dates)}
        return {"dataset": self.definition.name, "description": self.definition.description,
                "identity": self.profile.identity, "default_metric": self.definition.default_metric,
                "metrics": metrics, "note": "Operator-defined metrics. Null dates cannot enter dated analysis. "
                "Queries are read-only. Different metrics may have different units, grains, and filters."}

    def query(self, metric: str, start_date: str, end_date: str, group_by: str | None = None,
              interval: Literal["total", "day"] = "total", limit: int = 10):
        definition = self.metric(metric)
        if interval not in ("total", "day") or (group_by is not None and interval == "day"):
            raise ValueError("Use interval='total' or 'day'; do not combine a dimension and daily grouping.")
        if group_by is not None and (not isinstance(group_by, str) or group_by not in definition.dimensions):
            raise ValueError(f"Unknown dimension. Choose one of: {', '.join(definition.dimensions) or 'none'}.")
        limit = _bounded_integer(limit, "limit", 15)
        start_date, end_date, _ = self.date_range(start_date, end_date, 90 if interval == "day" else 366)
        rows, truncated, query = self._query(metric, start_date, end_date, group_by, limit, daily=interval == "day")
        quality = {"records": sum(row["records"] for row in rows),
                   "missing_values": sum(row["missing_values"] for row in rows), "scope": "returned rows"}
        if interval == "day":
            rows = [{"date": row["date"], "value": row["value"]} for row in rows]
        known = [row["value"] for row in rows if row["value"] is not None]
        return {"metric": metric, "label": definition.label, "definition": definition.description, "unit": definition.unit,
                "start_date": start_date, "end_date": end_date, "group_by": group_by, "interval": interval,
                "rows": rows, "truncated": truncated, "quality": quality, "query": query,
                "summary": {"observations": len(rows), "minimum": min(known) if known else None,
                            "maximum": max(known) if known else None},
                "note": "Missing numeric values are excluded from aggregates and counted. Missing or empty aggregates "
                "remain unknown; absent dates and omitted groups are not zero. Quality counts cover returned rows. "
                "Distinct counts can overlap across groups."}

    def compare(self, metric, start_date, end_date, baseline_start, baseline_end):
        definition = self.metric(metric)
        _, _, days = self.date_range(start_date, end_date)
        _, _, prior_days = self.date_range(baseline_start, baseline_end)
        current, _, current_sql = self._query(metric, start_date, end_date, round_values=False)
        baseline, _, baseline_sql = self._query(metric, baseline_start, baseline_end, round_values=False)
        now, prior = current[0]["value"], baseline[0]["value"]
        difference = now - prior if now is not None and prior is not None else None
        pct = difference / abs(prior) * 100 if difference is not None and prior else None
        current[0]["value"] = round(now, definition.precision) if now is not None else None
        baseline[0]["value"] = round(prior, definition.precision) if prior is not None else None
        return {"metric": metric, "label": definition.label, "unit": definition.unit,
                "current": {"start_date": start_date, "end_date": end_date, "days": days, **current[0]},
                "baseline": {"start_date": baseline_start, "end_date": baseline_end, "days": prior_days, **baseline[0]},
                "difference": round(difference, definition.precision) if difference is not None else None,
                "change_pct": round(pct, 2) if pct is not None else None,
                "direction": ("unknown" if difference is None else "higher" if difference > 0 else
                              "lower" if difference < 0 else "unchanged"),
                "queries": [current_sql, baseline_sql],
                "note": "Change is (current - baseline) / abs(baseline). A zero or unknown baseline has no percentage change. "
                + ("Periods have different lengths; totals are not time-normalized. " if days != prior_days else "")
                + "Ratio metrics use pooled numerators/denominators, not averages of daily averages. Causes remain unknown."}

    def anomalies(self, metric, start_date, end_date, direction="both", threshold_pct=30.0, limit=5):
        definition = self.metric(metric)
        start_date, end_date, _ = self.date_range(start_date, end_date)
        if direction not in ("both", "spikes", "drops"):
            raise ValueError("direction must be both, spikes, or drops.")
        if type(threshold_pct) not in (int, float) or not math.isfinite(threshold_pct) or not 0 < threshold_pct <= 10000:
            raise ValueError("threshold_pct must be a finite number greater than 0 and at most 10000.")
        limit = _bounded_integer(limit, "limit", 10)
        history_start = (date.fromisoformat(start_date) - timedelta(days=365)).isoformat()
        rows, _, query = self._query(metric, history_start, end_date, round_values=False, daily=True)
        histories, matches, skipped = defaultdict(list), [], 0
        for row in rows:
            day, value = date.fromisoformat(row["date"]), row["value"]
            prior = histories[day.weekday()][-4:]
            expected = median(prior) if len(prior) == 4 else None
            in_range = row["date"] >= start_date
            # Incomplete observations must not contaminate this baseline or be
            # called anomalies merely because source values are missing.
            usable = value is not None and row["missing_values"] == 0
            if in_range and (not usable or expected is None or expected <= 0):
                skipped += 1
            elif in_range:
                pct = (value - expected) / expected * 100
                if abs(pct) >= threshold_pct and (direction == "both" or
                     (direction == "spikes" and pct > 0) or (direction == "drops" and pct < 0)):
                    matches.append({"date": row["date"], "value": round(value, definition.precision),
                                    "expected": round(expected, definition.precision), "deviation_pct": round(pct, 2)})
            if usable:
                histories[day.weekday()].append(value)
        matches.sort(key=lambda row: (abs(row["deviation_pct"]), row["date"]), reverse=True)
        return {"metric": metric, "label": definition.label, "unit": definition.unit,
                "start_date": start_date, "end_date": end_date, "direction": direction, "threshold_pct": threshold_pct,
                "anomalies": matches[:limit], "total_matches": len(matches), "truncated": len(matches) > limit,
                "unassessed_observed_days": skipped, "query": query,
                "note": "Read-only calculation, not stored incidents. Baseline: median of four previous complete observed "
                "matching weekdays, using up to 365 days before the requested start. Non-positive or insufficient baselines "
                "and incomplete values are unassessed; missing dates are not filled. No causal conclusion is established."}

    def graph(self, metric, start_date, end_date, group_by=None, limit=10):
        self.date_range(start_date, end_date, 90)
        result = self.query(metric, start_date, end_date, group_by, "total" if group_by else "day", limit)
        x = "dimension" if group_by else "date"
        data = [{x: row[x], "value": row["value"]} for row in result["rows"]]
        known = [row["value"] for row in data if row["value"] is not None]
        if not known:
            data = []
        maximum = max(known) if known else None
        return {**{key: value for key, value in result.items() if key != "rows"}, "graph_type": "metric",
                "chart_type": "bar" if group_by or len(data) <= 1 else "line",
                "title": f"{result['label']}: {start_date} to {end_date}", "x": x, "y": "value",
                "y_label": f"{result['label']} ({result['unit']})", "data": data,
                "summary": {"observations": len(data), "minimum": min(known) if known else None, "maximum": maximum,
                            "maximum_at": [row[x] for row in data if row["value"] == maximum][:3] if known else []},
                "note": result["note"] + (" No matching known values; no graph is plotted." if not data else "")}


def model_tools(analytics: BusinessAnalytics):
    """The model can select registered operations and names, never identifiers or SQL."""
    def describe_business_dataset() -> dict:
        """List approved business metrics, units, dimensions, definitions, and date coverage."""
        return analytics.describe()

    def query_metric(metric: str, start_date: str, end_date: str, group_by: str | None = None,
                     interval: Literal["total", "day"] = "total", limit: int = 10) -> dict:
        """Query an approved metric: total, daily series, or dimension breakdown.

        Args:
            metric: Metric identifier from the dataset catalogue.
            start_date: Inclusive YYYY-MM-DD date.
            end_date: Inclusive YYYY-MM-DD date, at most 366 days; daily series at most 90 days.
            group_by: Approved dimension name, or omit; cannot combine with daily interval.
            interval: total or day.
            limit: Maximum dimension groups, from 1 to 15.
        """
        return analytics.query(metric, start_date, end_date, group_by, interval, limit)

    def compare_metric_periods(metric: str, start_date: str, end_date: str,
                               baseline_start: str, baseline_end: str) -> dict:
        """Compare approved metric values across two inclusive periods, each at most 366 days.

        Args:
            metric: Approved metric identifier.
            start_date: Current period first YYYY-MM-DD date.
            end_date: Current period last YYYY-MM-DD date.
            baseline_start: Baseline first YYYY-MM-DD date.
            baseline_end: Baseline last YYYY-MM-DD date.
        """
        return analytics.compare(metric, start_date, end_date, baseline_start, baseline_end)

    def find_metric_anomalies(metric: str, start_date: str, end_date: str,
                              direction: Literal["both", "spikes", "drops"] = "both",
                              threshold_pct: float = 30.0, limit: int = 5) -> dict:
        """Find unusual metric days against four previous matching weekdays, without writing data.

        Args:
            metric: Approved metric identifier.
            start_date: First YYYY-MM-DD date to inspect.
            end_date: Last YYYY-MM-DD date, at most 366 days inclusive.
            direction: both, spikes, or drops.
            threshold_pct: Absolute percentage deviation threshold, greater than zero and at most 10000.
            limit: Number of anomaly observations, from 1 to 10.
        """
        return analytics.anomalies(metric, start_date, end_date, direction, threshold_pct, limit)

    def create_metric_graph(metric: str, start_date: str, end_date: str,
                            group_by: str | None = None, limit: int = 10) -> dict:
        """Create a metric chart from actual data: daily line, or dimension bar chart.

        Args:
            metric: Approved metric identifier.
            start_date: First YYYY-MM-DD date.
            end_date: Last YYYY-MM-DD date, at most 90 days inclusive.
            group_by: Approved dimension for a bar chart, or omit for a daily chart.
            limit: Maximum groups, from 1 to 15.
        """
        return analytics.graph(metric, start_date, end_date, group_by, limit)

    return {fn.__name__: fn for fn in (describe_business_dataset, query_metric, compare_metric_periods,
                                      find_metric_anomalies, create_metric_graph)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a dataset configuration and inspect its approved metric catalogue.")
    parser.add_argument("config", help="Path to an operator-owned JSON configuration.")
    args = parser.parse_args()
    try:
        print(json.dumps(BusinessAnalytics(DatasetProfile.load(args.config)).describe(), indent=2))
    except (ValueError, OSError, RuntimeError, duckdb.Error) as exc:
        parser.exit(1, f"Dataset validation failed: {exc}\n")
