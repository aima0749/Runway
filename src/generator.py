"""Generate fictional invoice and expense data for Runway Radar scenarios."""

# duration T = days from issue_date to paid_date (model target, always >= 1).
# delay = days from due_date to paid_date (history feature and generator quantity).
# An unpaid invoice is censored: T = as_of - issue_date, E = 0. A paid invoice
# has E = 1. Never drop unpaid invoices.

from __future__ import annotations

import json
from calendar import monthrange
from datetime import date, timedelta
import numpy as np
import pandas as pd

from config import DATA_DIR, SEED

CLIENT_COLUMNS = ["client_id", "name", "segment"]
INVOICE_COLUMNS = ["invoice_id", "client_id", "issue_date", "due_date", "amount_pkr", "paid_date"]
EXPENSE_COLUMNS = ["expense_id", "category", "amount_pkr", "due_date", "recurrence"]
SEGMENT_DELAY = {"corporate": 17.0, "sme": 8.5, "individual": 3.5}
SEGMENTS = tuple(SEGMENT_DELAY)


def _month_start(value: date, offset: int) -> date:
    """Return the first day of a month offset from value."""
    index = value.year * 12 + value.month - 1 + offset
    return date(index // 12, index % 12 + 1, 1)


def _month_end(value: date) -> date:
    """Return the last day of value's month."""
    return date(value.year, value.month, monthrange(value.year, value.month)[1])


def _date_string(value: date) -> str:
    """Format a date using the schema's ISO date convention."""
    return value.isoformat()


def generate_business(
    seed: int = 0,
    n_clients: int = 25,
    months: int = 24,
    mode: str = "easy",
    start: str = "2024-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate deterministic fictional clients, invoices, and expenses."""
    if mode not in {"easy", "hard"}:
        raise ValueError("mode must be 'easy' or 'hard'")
    if n_clients < 0 or months < 1:
        raise ValueError("n_clients must be nonnegative and months must be positive")
    start_date = date.fromisoformat(start)
    end_date = _month_start(start_date, months)
    period_days = (end_date - start_date).days
    rng = np.random.default_rng(seed)
    clients: list[dict[str, object]] = []
    invoices: list[dict[str, object]] = []

    for client_num in range(n_clients):
        segment = SEGMENTS[int(rng.integers(len(SEGMENTS)))]
        trait: dict[str, object] = {
            "typical": SEGMENT_DELAY[segment] * rng.lognormal(0.0, 0.3),
            "sigma": float(rng.uniform(0.3, 0.6)),
            "late_chance": float(rng.uniform(0.03, 0.15)),
            "month_end": bool(rng.random() < 0.20),
            "size_k": float(rng.uniform(0.0, 0.25)),
            "change_date": start_date + timedelta(days=int(rng.integers(period_days))) if rng.random() < 0.40 else None,
            "change_factor": float(rng.uniform(0.8, 1.25)) if mode == "easy" else float(rng.choice([0.5, 2.0])),
        }
        client_id = f"C{client_num + 1:04d}"
        clients.append({"client_id": client_id, "name": f"Fictional Business {client_num + 1:04d}", "segment": segment})

        for month_idx in range(months):
            month = _month_start(start_date, month_idx)
            count = int(rng.poisson(1.5))
            for _ in range(count):
                issue = month + timedelta(days=int(rng.integers(monthrange(month.year, month.month)[1])))
                amount = float(rng.lognormal(10.8, 1.0))
                terms = int(rng.choice([15, 30, 45]))
                due = issue + timedelta(days=terms)
                typical = float(trait["typical"])
                if trait["change_date"] is not None and issue >= trait["change_date"]:
                    typical *= float(trait["change_factor"])

                if mode == "hard":
                    if rng.random() < 0.5:
                        delay = max(0.0, rng.normal(1.0, 1.5))
                    else:
                        delay = max(0.0, rng.normal(3.0 * typical, 0.20 * typical))
                    if rng.random() < float(trait["late_chance"]):
                        delay += float(rng.uniform(20.0, 50.0))
                else:
                    delay = max(0.0, typical * rng.lognormal(-0.5 * float(trait["sigma"]) ** 2, float(trait["sigma"])))
                    if rng.random() < float(trait["late_chance"]):
                        delay += float(rng.uniform(20.0, 50.0))
                delay *= float(np.exp(float(trait["size_k"]) * np.log(amount / 50_000.0)))

                paid = due + timedelta(days=int(round(delay)))
                if trait["month_end"] and paid < _month_end(paid):
                    paid = _month_end(paid)
                paid = max(paid, issue + timedelta(days=1))
                invoice_id = f"INV{len(invoices) + 1:07d}"
                invoices.append({
                    "invoice_id": invoice_id,
                    "client_id": client_id,
                    "issue_date": _date_string(issue),
                    "due_date": _date_string(due),
                    "amount_pkr": round(amount, 2),
                    "paid_date": _date_string(paid),
                })

    # Estimate a plausible monthly run rate from the generated monthly invoice volume.
    invoice_total = sum(float(row["amount_pkr"]) for row in invoices)
    monthly_revenue = invoice_total / months if months else 0.0
    monthly_cost = monthly_revenue * 0.65
    expenses: list[dict[str, object]] = []
    # A monthly row is a recurring template consumed by simulate_cash. Emit
    # one template per category instead of one duplicate template per month.
    month = _month_start(start_date, 0)
    for category, day, share in (("rent", 1, 0.20), ("salaries", 20, 0.55), ("suppliers", 25, 0.25)):
        due = date(month.year, month.month, min(day, monthrange(month.year, month.month)[1]))
        expenses.append({
            "expense_id": f"E{len(expenses) + 1:06d}",
            "category": category,
            "amount_pkr": round(monthly_cost * share, 2),
            "due_date": _date_string(due),
            "recurrence": "monthly",
        })

    return (
        pd.DataFrame(clients, columns=CLIENT_COLUMNS),
        pd.DataFrame(invoices, columns=INVOICE_COLUMNS),
        pd.DataFrame(expenses, columns=EXPENSE_COLUMNS),
    )


def censor_at(invoices: pd.DataFrame, as_of: str | date) -> pd.DataFrame:
    """Return invoices observable by as_of while retaining their true paid dates."""
    cutoff = pd.Timestamp(as_of)
    result = invoices.copy()
    issue_dates = pd.to_datetime(result["issue_date"])
    result = result.loc[issue_dates <= cutoff].copy()
    result["true_paid_date"] = result["paid_date"]
    paid_dates = pd.to_datetime(result["paid_date"])
    result.loc[paid_dates > cutoff, "paid_date"] = pd.NaT
    return result.reset_index(drop=True)


def main() -> None:
    """Write fixed-seed easy and hard datasets under the configured data path."""
    for mode in ("easy", "hard"):
        clients, invoices, expenses = generate_business(seed=SEED, mode=mode)
        output = DATA_DIR / f"synthetic_{mode}"
        output.mkdir(parents=True, exist_ok=True)
        clients.to_csv(output / "clients.csv", index=False)
        invoices.to_csv(output / "invoices.csv", index=False)
        expenses.to_csv(output / "expenses.csv", index=False)
        settings = {
            "as_of_date": "2025-12-31",
            "opening_cash_pkr": round(float(expenses["amount_pkr"].sum()) * 2, 2),
            "min_buffer_pkr": round(float(expenses["amount_pkr"].sum()), 2),
        }
        (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
