"""Operator-owned business definitions, never SQL or instructions from the model."""
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")]
Reference = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}\.[A-Za-z_][A-Za-z0-9_]{0,63}$")]


def quoted(name: str) -> str:
    """Identifiers come only from validated configuration, never model SQL."""
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
        raise ValueError(f"Invalid SQL identifier: {name!r}.")
    return f'"{name}"'


def column_sql(reference: str) -> str:
    parts = reference.split(".")
    if len(parts) != 2:
        raise ValueError("Columns must use a configured table alias and column name.")
    return ".".join(quoted(part) for part in parts)


class Definition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class Table(Definition):
    table: Identifier
    grain: list[Identifier] = Field(min_length=1, max_length=4)


class Join(Definition):
    left: Reference
    right: Reference
    kind: Literal["inner", "left"] = "left"


class Term(Definition):
    aggregation: Literal["sum", "average", "count", "count_distinct"]
    column: Reference | None = None

    @model_validator(mode="after")
    def check_column(self):
        if (self.aggregation == "count") != (self.column is None):
            raise ValueError("count requires no column; other aggregations require a column.")
        return self


class Calculation(Definition):
    aggregation: Literal["sum", "average", "count", "count_distinct", "ratio"]
    column: Reference | None = None
    numerator: Term | None = None
    denominator: Term | None = None

    @model_validator(mode="after")
    def check_calculation(self):
        if self.aggregation == "ratio":
            if self.column is not None or self.numerator is None or self.denominator is None:
                raise ValueError("ratio requires numerator and denominator terms, without a column.")
        else:
            Term(aggregation=self.aggregation, column=self.column)
            if self.numerator is not None or self.denominator is not None:
                raise ValueError("Only ratio calculations accept numerator and denominator.")
        return self

    def terms(self) -> list[Term]:
        if self.aggregation == "ratio":
            return [self.numerator, self.denominator]
        return [Term(aggregation=self.aggregation, column=self.column)]


class Filter(Definition):
    column: Reference
    value: str | int | float | bool | None

    @model_validator(mode="after")
    def check_value(self):
        if isinstance(self.value, str) and len(self.value) > 100:
            raise ValueError("Filter text must be at most 100 characters.")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("Filter numbers must be finite.")
        return self


class Metric(Definition):
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=400)
    source: Identifier
    joins: list[Identifier] = Field(default_factory=list, max_length=8)
    date: Reference
    calculation: Calculation
    unit: str = Field(min_length=1, max_length=40)
    precision: int = Field(default=2, ge=0, le=6)
    filters: list[Filter] = Field(default_factory=list, max_length=8)
    dimensions: dict[Identifier, Reference] = Field(default_factory=dict, max_length=8)


class Dataset(Definition):
    version: Literal[1]
    id: Identifier
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=400)
    database: str = Field(min_length=1, max_length=500)
    default_metric: Identifier
    tables: dict[Identifier, Table] = Field(min_length=1, max_length=8)
    joins: dict[Identifier, Join] = Field(default_factory=dict, max_length=8)
    metrics: dict[Identifier, Metric] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def check_references(self):
        if self.default_metric not in self.metrics:
            raise ValueError("default_metric must name a configured metric.")
        if "://" in self.database or self.database == ":memory:":
            raise ValueError("database must be a local persistent DuckDB file.")
        for alias, table in self.tables.items():
            if len(set(table.grain)) != len(table.grain):
                raise ValueError(f"Duplicate grain columns for {alias}.")
        if len({alias.casefold() for alias in self.tables}) != len(self.tables):
            raise ValueError("Table aliases must also be unique without regard to case.")
        for name, join in self.joins.items():
            left, right = join.left.split("."), join.right.split(".")
            if left[0] not in self.tables or right[0] not in self.tables:
                raise ValueError(f"Unknown table alias in join {name}.")
            if left[0] == right[0] or self.tables[right[0]].grain != [right[1]]:
                raise ValueError(f"Join {name} must target a different table's single-column unique grain.")
        for name, metric in self.metrics.items():
            if metric.source not in self.tables:
                raise ValueError(f"Unknown source for metric {name}.")
            available = {metric.source}
            for join_name in metric.joins:
                if join_name not in self.joins:
                    raise ValueError(f"Unknown join {join_name} for metric {name}.")
                join = self.joins[join_name]
                left, right = join.left.split(".")[0], join.right.split(".")[0]
                if left not in available or right in available:
                    raise ValueError(f"Joins for {name} must form an ordered, non-repeating chain from its source.")
                available.add(right)
            references = [metric.date, *metric.dimensions.values(), *(f.column for f in metric.filters),
                          *(t.column for t in metric.calculation.terms() if t.column)]
            if any(ref.split(".")[0] not in available for ref in references):
                raise ValueError(f"Metric {name} references a table outside its approved joins.")
        return self


class DatasetProfile:
    """Configuration identity isolates sessions when definitions or data change."""
    def __init__(self, definition: Dataset, path: Path):
        self.definition = definition
        self.path = path.resolve()
        database = Path(definition.database).expanduser()
        self.database = (self.path.parent / database).resolve()
        if not self.database.is_file():
            raise FileNotFoundError(f"Configured DuckDB file does not exist: {self.database}. "
                                    "Prepare that database first; configuration never creates or overwrites it.")
        stat = self.database.stat()
        self._database_stamp = (stat.st_size, stat.st_mtime_ns)
        identity = [definition.model_dump(mode="json"), str(self.database), stat.st_size, stat.st_mtime_ns]
        self.identity = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def check_unchanged(self):
        stat = self.database.stat()
        if (stat.st_size, stat.st_mtime_ns) != self._database_stamp:
            raise RuntimeError("The configured data changed. Reload its profile and start a new conversation.")

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if path.stat().st_size > 65536:
            raise ValueError("Dataset configuration exceeds the 64 KiB limit.")
        try:
            definition = Dataset.model_validate_json(path.read_text())
        except ValueError as exc:
            raise ValueError(f"Invalid dataset configuration {path.name}: {exc}") from exc
        return cls(definition, path)
