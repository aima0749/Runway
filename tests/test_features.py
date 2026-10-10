"""Tests for leakage-safe invoice features and survival targets."""

import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, build_features


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a small fixed invoice history with late and censored rows."""
    clients = pd.DataFrame({
        "client_id": ["A", "B"],
        "segment": ["sme", "corporate"],
    })
    invoices = pd.DataFrame([
        # A1 is paid before A2's issue and is valid history for A2.
        {"invoice_id": "A1", "client_id": "A", "issue_date": "2024-01-01", "due_date": "2024-01-10", "amount_pkr": 1000, "paid_date": "2024-01-20"},
        # A2 is paid after A3's issue, so it must not be used for A3.
        {"invoice_id": "A2", "client_id": "A", "issue_date": "2024-01-15", "due_date": "2024-01-25", "amount_pkr": 2000, "paid_date": "2024-02-10"},
        {"invoice_id": "A3", "client_id": "A", "issue_date": "2024-02-01", "due_date": "2024-02-11", "amount_pkr": 3000, "paid_date": None},
        {"invoice_id": "A4", "client_id": "A", "issue_date": "2024-02-15", "due_date": "2024-02-25", "amount_pkr": 4000, "paid_date": "2024-03-01"},
        # A later invoice from another client must not enter A's history.
        {"invoice_id": "B1", "client_id": "B", "issue_date": "2024-01-01", "due_date": "2024-01-11", "amount_pkr": 5000, "paid_date": "2024-01-12"},
    ])
    return invoices, clients


def test_history_uses_only_payments_on_or_before_invoice_issue() -> None:
    invoices, clients = _inputs()
    X, _, _, _ = build_features(invoices, clients, "2024-03-31")
    # A3 can use A1 but not A2, whose payment occurred after A3 was issued.
    assert X.loc[2, "n_prior"] == 1
    assert X.loc[2, "last_delay"] == 10
    # A2 cannot use itself, even though it is paid later.
    assert X.loc[1, "n_prior"] == 0


def test_later_payment_changes_do_not_change_earlier_features() -> None:
    invoices, clients = _inputs()
    before, _, _, _ = build_features(invoices, clients, "2024-03-31")
    changed = invoices.copy()
    changed.loc[changed["invoice_id"] == "A4", "paid_date"] = "2024-03-20"
    after, _, _, _ = build_features(changed, clients, "2024-03-31")
    earlier = invoices["issue_date"] < "2024-02-15"
    pd.testing.assert_frame_equal(before.loc[earlier], after.loc[earlier])


def test_unpaid_invoice_is_right_censored_at_as_of() -> None:
    invoices, clients = _inputs()
    _, T, E, _ = build_features(invoices, clients, "2024-02-20")
    assert E[2] == 0
    assert T[2] == 19


def test_cold_start_is_true_when_fewer_than_three_prior_payments() -> None:
    invoices, clients = _inputs()
    _, _, _, aligned = build_features(invoices, clients, "2024-03-31")
    assert aligned["cold_start"].tolist() == [True, True, True, True, True]


def test_feature_columns_match_constant_in_order() -> None:
    invoices, clients = _inputs()
    X, T, E, aligned = build_features(invoices, clients, "2024-03-31")
    assert X.columns.tolist() == FEATURE_COLS
    assert X.columns[-3:].tolist() == ["seg_corporate", "seg_sme", "seg_individual"]
    assert len(X) == len(T) == len(E) == len(aligned)


def test_missing_history_features_stay_nan() -> None:
    invoices, clients = _inputs()
    X, _, _, _ = build_features(invoices, clients, "2024-03-31")
    history_features = ["hist_mean_delay", "hist_std_delay", "hist_share_late_gt7", "last_delay"]
    assert X.loc[1, history_features].isna().all()
    assert np.isnan(X.loc[1, "hist_mean_delay"])
