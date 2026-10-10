"""Load and validate CSV business data using the repository schema."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.schemas import Dataset, Settings

TABLE_COLUMNS = {
    "clients": ["client_id", "name", "segment"],
    "invoices": ["invoice_id", "client_id", "issue_date", "due_date", "amount_pkr", "paid_date"],
    "expenses": ["expense_id", "category", "amount_pkr", "due_date", "recurrence"],
}


def _read_settings(settings: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Read settings from a JSON file or mapping."""
    if isinstance(settings, Mapping):
        return dict(settings)
    return json.loads(Path(settings).read_text(encoding="utf-8"))


def load_and_validate(
    files: Mapping[str, str | Path] | str | Path,
    settings: str | Path | Mapping[str, Any],
) -> tuple[Dataset, list[str]]:
    """Load CSV tables and settings, returning a Dataset and validation errors."""
    if isinstance(files, (str, Path)):
        root = Path(files)
        paths = {name: root / f"{name}.csv" for name in TABLE_COLUMNS}
    else:
        paths = dict(files)

    errors: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    for name, columns in TABLE_COLUMNS.items():
        path = paths.get(name)
        if path is None or not Path(path).is_file():
            errors.append(f"Missing {name}.csv")
            tables[name] = pd.DataFrame(columns=columns)
            continue
        try:
            table = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            errors.append(f"Could not read {name}.csv: {exc}")
            tables[name] = pd.DataFrame(columns=columns)
            continue
        missing = [column for column in columns if column not in table.columns]
        if missing:
            errors.append(f"{name}.csv missing columns: {', '.join(missing)}")
            tables[name] = pd.DataFrame(columns=columns)
            continue
        if table.columns.tolist() != columns:
            errors.append(f"{name}.csv columns must be exactly: {', '.join(columns)}")
            table = table.loc[:, columns]
        tables[name] = table

    try:
        raw_settings = _read_settings(settings)
        as_of = str(raw_settings["as_of_date"])
        pd.Timestamp(as_of)
        opening_cash = float(raw_settings["opening_cash_pkr"])
        min_buffer = float(raw_settings.get("min_buffer_pkr", 0.0))
        if opening_cash < 0 or min_buffer < 0:
            raise ValueError("cash settings must be nonnegative")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid settings: {exc}")
        as_of, opening_cash, min_buffer = "1970-01-01", 0.0, 0.0

    for table_name in ("invoices", "expenses"):
        date_columns = ["issue_date", "due_date"] if table_name == "invoices" else ["due_date"]
        if table_name == "invoices" and "paid_date" in tables[table_name]:
            date_columns.append("paid_date")
        for column in date_columns:
            try:
                pd.to_datetime(tables[table_name][column], errors="raise")
            except (ValueError, TypeError):
                errors.append(f"{table_name}.csv has invalid {column} values")

    if not tables["clients"].empty and "segment" in tables["clients"]:
        allowed = {"corporate", "sme", "individual"}
        if not set(tables["clients"]["segment"].dropna()).issubset(allowed):
            errors.append("clients.csv segment must be corporate, sme, or individual")
    if not tables["expenses"].empty and "recurrence" in tables["expenses"]:
        allowed_recurrence = {"once", "monthly", "weekly"}
        if not set(tables["expenses"]["recurrence"].dropna()).issubset(allowed_recurrence):
            errors.append("expenses.csv recurrence must be once, monthly, or weekly")

    dataset = Dataset(
        clients=tables["clients"],
        invoices=tables["invoices"],
        expenses=tables["expenses"],
        settings=Settings(
            as_of_date=as_of,
            opening_cash_pkr=opening_cash,
            min_buffer_pkr=min_buffer,
        ),
    )
    return dataset, errors
