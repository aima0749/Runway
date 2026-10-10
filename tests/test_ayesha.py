"""Checks for the fictional Ayesha Studio scripted dataset."""

import pandas as pd

from src.data import load_and_validate
from src.scenario_ayesha import AS_OF_DATE, make_ayesha_dataset, write_ayesha_dataset


def test_ayesha_files_load_and_have_five_large_open_invoices(tmp_path) -> None:
    write_ayesha_dataset(tmp_path)
    dataset, errors = load_and_validate(tmp_path, tmp_path / "settings.json")

    assert errors == []
    assert (tmp_path / "clients.csv").is_file()
    assert (tmp_path / "invoices.csv").is_file()
    assert (tmp_path / "expenses.csv").is_file()
    assert (tmp_path / "settings.json").is_file()
    assert (tmp_path / "README.md").is_file()
    assert len(dataset.clients) == 11
    assert len(dataset.expenses) == 3
    assert set(dataset.expenses["category"]) == {"rent", "salaries", "suppliers"}
    assert dataset.expenses["recurrence"].eq("monthly").all()

    paid = pd.to_datetime(dataset.invoices["paid_date"])
    open_invoices = dataset.invoices.loc[paid > pd.Timestamp(AS_OF_DATE)]
    large_open = open_invoices.loc[open_invoices["amount_pkr"] >= 50_000]
    assert len(large_open) == 5
    assert 750_000 <= large_open["amount_pkr"].sum() <= 850_000


def test_ayesha_build_is_deterministic() -> None:
    first = make_ayesha_dataset()
    second = make_ayesha_dataset()
    pd.testing.assert_frame_equal(first.clients, second.clients)
    pd.testing.assert_frame_equal(first.invoices, second.invoices)
    pd.testing.assert_frame_equal(first.expenses, second.expenses)
    assert first.settings == second.settings
