"""Build leakage-safe invoice features and survival targets."""

# duration T = days from issue_date to paid_date (model target, always >= 1).
# delay = days from due_date to paid_date (used for history features).
# An unpaid invoice is censored: T = as_of - issue_date, E = 0. A paid invoice
# has E = 1. Never drop unpaid invoices.

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "n_prior",
    "hist_mean_delay",
    "hist_std_delay",
    "hist_share_late_gt7",
    "last_delay",
    "log_amount",
    "terms",
    "due_day_ge_25",
    "seg_corporate",
    "seg_sme",
    "seg_individual",
]


def build_features(
    invoices: pd.DataFrame,
    clients: pd.DataFrame,
    as_of: str | pd.Timestamp,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build invoice features and right-censored duration targets."""
    required_invoice_cols = {"invoice_id", "client_id", "issue_date", "due_date", "amount_pkr", "paid_date"}
    required_client_cols = {"client_id", "segment"}
    missing_invoices = required_invoice_cols.difference(invoices.columns)
    missing_clients = required_client_cols.difference(clients.columns)
    if missing_invoices:
        raise ValueError(f"invoices missing columns: {sorted(missing_invoices)}")
    if missing_clients:
        raise ValueError(f"clients missing columns: {sorted(missing_clients)}")
    if clients["client_id"].duplicated().any():
        raise ValueError("clients must contain one row per client_id")

    aligned = invoices.copy()
    issue = pd.to_datetime(aligned["issue_date"], errors="raise")
    due = pd.to_datetime(aligned["due_date"], errors="raise")
    paid = pd.to_datetime(aligned["paid_date"], errors="coerce")
    cutoff = pd.Timestamp(as_of)
    amount = pd.to_numeric(aligned["amount_pkr"], errors="raise").to_numpy(dtype=float)
    if np.any(amount <= 0):
        raise ValueError("amount_pkr must be positive to compute log_amount")

    client_segments = clients.set_index("client_id")["segment"]
    segment = aligned["client_id"].map(client_segments)
    feature_rows: list[dict[str, float]] = []
    for position in range(len(aligned)):
        issue_date = issue.iloc[position]
        current_client = aligned["client_id"].iloc[position]
        history_mask = (
            (aligned["client_id"] == current_client)
            & paid.notna()
            & (paid <= cutoff)
            & (paid <= issue_date)
        )
        history_mask.iloc[position] = False
        history_positions = np.flatnonzero(history_mask.to_numpy())
        history_delays = (paid.iloc[history_positions] - due.iloc[history_positions]).dt.days.to_numpy(dtype=float)
        n_prior = len(history_delays)

        if n_prior:
            most_recent = np.argmax(paid.iloc[history_positions].astype("int64").to_numpy())
            last_delay = float(history_delays[most_recent])
            mean_delay = float(np.mean(history_delays))
            std_delay = float(np.std(history_delays, ddof=0))
            share_late = float(np.mean(history_delays > 7.0))
        else:
            last_delay = mean_delay = std_delay = share_late = np.nan

        feature_rows.append({
            "n_prior": float(n_prior),
            "hist_mean_delay": mean_delay,
            "hist_std_delay": std_delay,
            "hist_share_late_gt7": share_late,
            "last_delay": last_delay,
            "log_amount": float(np.log(amount[position])),
            "terms": float((due.iloc[position] - issue_date).days),
            "due_day_ge_25": float(due.iloc[position].day >= 25),
            "seg_corporate": float(segment.iloc[position] == "corporate"),
            "seg_sme": float(segment.iloc[position] == "sme"),
            "seg_individual": float(segment.iloc[position] == "individual"),
        })

    X = pd.DataFrame(feature_rows, columns=FEATURE_COLS, index=aligned.index)
    aligned["cold_start"] = X["n_prior"].lt(3).to_numpy(dtype=bool)

    event_observed = paid.notna() & (paid <= cutoff)
    duration = np.where(
        event_observed.to_numpy(),
        (paid - issue).dt.days.to_numpy(dtype=float),
        (cutoff - issue).dt.days.to_numpy(dtype=float),
    )
    T = np.maximum(duration, 1).astype(int)
    E = event_observed.to_numpy(dtype=int)
    return X, T, E, aligned
