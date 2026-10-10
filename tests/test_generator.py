"""Tests for deterministic fictional business data generation."""

import pandas as pd

from src.generator import censor_at, generate_business


def _tables_equal(left: tuple[pd.DataFrame, ...], right: tuple[pd.DataFrame, ...]) -> bool:
    """Compare each generated table, including row order and values."""
    return all(a.equals(b) for a, b in zip(left, right))


def test_same_seed_is_identical_and_different_seed_differs() -> None:
    first = generate_business(seed=17, n_clients=30, months=6)
    assert _tables_equal(first, generate_business(seed=17, n_clients=30, months=6))
    assert not _tables_equal(first, generate_business(seed=18, n_clients=30, months=6))


def test_columns_and_payment_dates_match_contract() -> None:
    clients, invoices, expenses = generate_business(seed=3, n_clients=20, months=5)
    assert clients.columns.tolist() == ["client_id", "name", "segment"]
    assert invoices.columns.tolist() == ["invoice_id", "client_id", "issue_date", "due_date", "amount_pkr", "paid_date"]
    assert expenses.columns.tolist() == ["expense_id", "category", "amount_pkr", "due_date", "recurrence"]
    assert set(clients["segment"]).issubset({"corporate", "sme", "individual"})
    assert set(expenses["recurrence"]).issubset({"once", "monthly", "weekly"})
    issue = pd.to_datetime(invoices["issue_date"])
    paid = pd.to_datetime(invoices["paid_date"])
    assert (paid >= issue + pd.Timedelta(days=1)).all()


def test_segment_delay_means_are_ordered() -> None:
    _, invoices, _ = generate_business(seed=24, n_clients=1800, months=8)
    joined = invoices.assign(delay=(pd.to_datetime(invoices.paid_date) - pd.to_datetime(invoices.due_date)).dt.days)
    # Segment labels are recoverable from stable client assignment for this seed.
    clients, _, _ = generate_business(seed=24, n_clients=1800, months=8)
    joined = joined.merge(clients[["client_id", "segment"]], on="client_id")
    means = joined.groupby("segment").delay.mean()
    assert means["corporate"] > means["sme"] > means["individual"]


def test_hard_mode_has_two_delay_bands() -> None:
    _, invoices, _ = generate_business(seed=33, n_clients=400, months=8, mode="hard")
    delay = (pd.to_datetime(invoices.paid_date) - pd.to_datetime(invoices.due_date)).dt.days
    low_share = float((delay <= 5).mean())
    high_share = float((delay >= 12).mean())
    assert 0.30 < low_share < 0.70
    assert high_share > 0.25


def test_censor_at_preserves_truth_and_hides_future_payments() -> None:
    _, invoices, _ = generate_business(seed=9, n_clients=80, months=10, start="2024-01-01")
    cutoff = "2024-07-15"
    observed = censor_at(invoices, cutoff)
    assert (pd.to_datetime(observed.issue_date) <= pd.Timestamp(cutoff)).all()
    visible_paid = pd.to_datetime(observed.paid_date).dropna()
    assert (visible_paid <= pd.Timestamp(cutoff)).all()
    truth = pd.to_datetime(observed.true_paid_date)
    hidden = truth > pd.Timestamp(cutoff)
    assert observed.loc[hidden, "paid_date"].isna().all()
    assert (pd.to_datetime(observed.loc[hidden, "true_paid_date"]) > pd.Timestamp(cutoff)).all()
