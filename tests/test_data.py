"""Tests for loading repository-shaped business data."""

import json
from pathlib import Path

import pandas as pd

from src.data import load_and_validate
from src.schemas import Dataset, Settings


def _write_tables(directory: Path) -> dict[str, Path]:
    """Write small CSV fixtures with the schema's exact column order."""
    tables = {
        "clients": pd.DataFrame(
            [{"client_id": "C001", "name": "Fictional Client", "segment": "sme"}],
            columns=["client_id", "name", "segment"],
        ),
        "invoices": pd.DataFrame(
            [{
                "invoice_id": "I001",
                "client_id": "C001",
                "issue_date": "2025-01-01",
                "due_date": "2025-01-31",
                "amount_pkr": 1000.0,
                "paid_date": None,
            }],
            columns=["invoice_id", "client_id", "issue_date", "due_date", "amount_pkr", "paid_date"],
        ),
        "expenses": pd.DataFrame(
            [{
                "expense_id": "E001",
                "category": "rent",
                "amount_pkr": 500.0,
                "due_date": "2025-02-01",
                "recurrence": "monthly",
            }],
            columns=["expense_id", "category", "amount_pkr", "due_date", "recurrence"],
        ),
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def test_load_and_validate_directory_returns_dataset_and_settings(tmp_path) -> None:
    _write_tables(tmp_path)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"as_of_date": "2025-02-01", "opening_cash_pkr": 2500, "min_buffer_pkr": 200}),
        encoding="utf-8",
    )

    dataset, errors = load_and_validate(tmp_path, settings_path)

    assert errors == []
    assert isinstance(dataset, Dataset)
    assert isinstance(dataset.settings, Settings)
    assert dataset.settings.as_of_date == "2025-02-01"
    assert dataset.settings.opening_cash_pkr == 2500
    assert dataset.settings.min_buffer_pkr == 200
    assert dataset.clients.columns.tolist() == ["client_id", "name", "segment"]
    assert dataset.invoices.columns.tolist() == [
        "invoice_id", "client_id", "issue_date", "due_date", "amount_pkr", "paid_date",
    ]
    assert dataset.expenses.columns.tolist() == [
        "expense_id", "category", "amount_pkr", "due_date", "recurrence",
    ]
    assert pd.isna(dataset.invoices.loc[0, "paid_date"])


def test_load_and_validate_accepts_table_path_mapping_and_settings_mapping(tmp_path) -> None:
    paths = _write_tables(tmp_path)
    settings = {"as_of_date": "2025-02-01", "opening_cash_pkr": 2500}

    dataset, errors = load_and_validate(paths, settings)

    assert errors == []
    assert isinstance(dataset, Dataset)
    assert isinstance(dataset.settings, Settings)
    assert dataset.settings.min_buffer_pkr == 0.0
