"""Create and inspect the scripted fictional Ayesha Studio demo scenario."""

# duration T = days from issue_date to paid_date (model target, always >= 1).
# delay = days from due_date to paid_date (history feature and scenario trait).
# An unpaid invoice is censored: T = as_of - issue_date, E = 0. A paid invoice
# has E = 1. Never drop unpaid invoices.

from __future__ import annotations

import inspect
import json
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON_DAYS, N_SIMS, SEED
from src.delay_model import build_pipeline
from src.generator import CLIENT_COLUMNS, EXPENSE_COLUMNS, INVOICE_COLUMNS
from src.schemas import Dataset, Settings
from src.simulate import simulate_cash

AS_OF_DATE = "2025-02-15"
OUTPUT_DIR = DATA_DIR / "ayesha_studio"
CLIENTS = [
    ("AY001", "Fictional Northstar Retail", "corporate"),
    ("AY002", "Fictional Meridian Goods", "corporate"),
    ("AY003", "Fictional Paper Kite Agency", "corporate"),
    ("AY004", "Fictional Small Studio Client 1", "sme"),
    ("AY005", "Fictional Small Studio Client 2", "sme"),
    ("AY006", "Fictional Small Studio Client 3", "sme"),
    ("AY007", "Fictional Local Shop 1", "sme"),
    ("AY008", "Fictional Local Shop 2", "sme"),
    ("AY009", "Fictional Individual Client 1", "individual"),
    ("AY010", "Fictional Individual Client 2", "individual"),
    ("AY011", "Fictional Individual Client 3", "individual"),
]
HISTORICAL_DELAY = {"AY001": 21, "AY002": 28, "AY003": 13}
SMALL_CLIENTS = tuple(client[0] for client in CLIENTS[3:])
OPEN_ROWS = (
    ("AY003", "2025-01-15", 30, 500_000.0, 13),
    ("AY001", "2025-01-20", 30, 100_000.0, 21),
    ("AY002", "2025-01-21", 30, 80_000.0, 28),
    ("AY002", "2025-01-25", 30, 70_000.0, 28),
    ("AY003", "2025-01-25", 30, 50_000.0, 13),
)
EXPENSE_AMOUNTS = {"rent": 60_000.0, "salaries": 300_000.0, "suppliers": 60_000.0}
EXPENSE_DAYS = {"rent": 1, "salaries": 20, "suppliers": 25}


def _month_start(value: date, offset: int) -> date:
    """Return the first of the requested month in the scenario history."""
    index = value.year * 12 + value.month - 1 + offset
    return date(index // 12, index % 12 + 1, 1)


def _payment_delay(client_id: str, month_index: int, invoice_index: int = 0) -> int:
    """Return a fixed scenario delay for one historical invoice."""
    if client_id in HISTORICAL_DELAY:
        return HISTORICAL_DELAY[client_id]
    return 1 + ((month_index + invoice_index + int(client_id[-1])) % 3)


def make_ayesha_dataset() -> Dataset:
    """Build the same 11-client, 12-month fictional dataset on every call."""
    clients = pd.DataFrame(CLIENTS, columns=CLIENT_COLUMNS)
    invoices: list[dict[str, object]] = []
    invoice_number = 1

    for month_index in range(12):
        month = _month_start(date(2024, 1, 1), month_index)
        for client_index, (client_id, _, _) in enumerate(CLIENTS):
            # The three large accounts have monthly history. Small accounts
            # receive one monthly invoice and an additional invoice every third month.
            issue_day = 1 if client_id in HISTORICAL_DELAY else 1 + (client_index + month_index) % 5
            issue = month + timedelta(days=issue_day - 1)
            amount = (
                float(92_000 + 4_000 * ((client_index + month_index) % 6))
                if client_id in HISTORICAL_DELAY
                else float(42_000 + 3_000 * ((client_index + month_index) % 6))
            )
            history_count = 2 if client_id in SMALL_CLIENTS[:3] and month_index % 3 == 1 else 1
            for invoice_in_month in range(history_count):
                current_issue = issue + timedelta(days=9 * invoice_in_month)
                current_amount = amount if invoice_in_month == 0 else round(amount * 0.7, 2)
                terms = 15
                due = current_issue + timedelta(days=terms)
                delay = _payment_delay(client_id, month_index, invoice_in_month)
                paid = due + timedelta(days=delay)
                invoices.append({
                    "invoice_id": f"AYINV{invoice_number:05d}",
                    "client_id": client_id,
                    "issue_date": current_issue.isoformat(),
                    "due_date": due.isoformat(),
                    "amount_pkr": current_amount,
                    "paid_date": paid.isoformat(),
                })
                invoice_number += 1

    for client_id, issue_text, terms, amount, delay in OPEN_ROWS:
        issue = date.fromisoformat(issue_text)
        due = issue + timedelta(days=terms)
        invoices.append({
            "invoice_id": f"AYOPEN{invoice_number:02d}",
            "client_id": client_id,
            "issue_date": issue.isoformat(),
            "due_date": due.isoformat(),
            "amount_pkr": amount,
            "paid_date": (due + timedelta(days=delay)).isoformat(),
        })
        invoice_number += 1

    invoice_table = pd.DataFrame(invoices, columns=INVOICE_COLUMNS)
    expenses: list[dict[str, object]] = []
    expense_number = 1
    # Include a year of scenario history and enough known recurring bills to
    # cover the model's configured forecast horizon.
    for month_index in range(16):
        month = _month_start(date(2024, 1, 1), month_index)
        for category, due_day in EXPENSE_DAYS.items():
            due = date(month.year, month.month, min(due_day, monthrange(month.year, month.month)[1]))
            expenses.append({
                "expense_id": f"AYEXP{expense_number:04d}",
                "category": category,
                "amount_pkr": EXPENSE_AMOUNTS[category],
                "due_date": due.isoformat(),
                "recurrence": "monthly",
            })
            expense_number += 1

    expense_table = pd.DataFrame(expenses, columns=EXPENSE_COLUMNS)
    opening_cash = _paper_projection_opening_cash(
        invoice_table,
        expense_table,
        pd.Timestamp(AS_OF_DATE),
    )
    return Dataset(
        clients=clients,
        invoices=invoice_table,
        expenses=expense_table,
        settings=Settings(
            as_of_date=AS_OF_DATE,
            opening_cash_pkr=opening_cash,
            min_buffer_pkr=0.0,
        ),
    )


def _future_expense_rows(expenses: pd.DataFrame, as_of: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Select known expense dates inside the chosen projection horizon."""
    due = pd.to_datetime(expenses["due_date"])
    end = as_of + pd.Timedelta(days=horizon)
    return expenses.loc[(due > as_of) & (due <= end)].copy()


def _cash_path(
    dataset: Dataset,
    payment_dates: np.ndarray,
    horizon: int = HORIZON_DAYS,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Project daily cash from as-of using invoice receipts and expenses."""
    as_of = pd.Timestamp(dataset.settings.as_of_date)
    dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon, freq="D")
    net = np.zeros(horizon, dtype=float)
    open_invoices = dataset.invoices.loc[
        pd.to_datetime(dataset.invoices["paid_date"]) > as_of
    ].reset_index(drop=True)
    for amount, payment_date in zip(open_invoices["amount_pkr"].to_numpy(dtype=float), payment_dates):
        day = pd.Timestamp(payment_date).normalize()
        offset = (day - dates[0]).days
        if 0 <= offset < horizon:
            net[offset] += amount
    future_expenses = _future_expense_rows(dataset.expenses, as_of, horizon)
    for row in future_expenses.itertuples():
        offset = (pd.Timestamp(row.due_date).normalize() - dates[0]).days
        if 0 <= offset < horizon:
            net[offset] -= float(row.amount_pkr)
    cash = dataset.settings.opening_cash_pkr + np.cumsum(net)
    return dates, cash


def _paper_projection_opening_cash(
    invoices: pd.DataFrame,
    expenses: pd.DataFrame,
    as_of: pd.Timestamp,
) -> float:
    """Set cash to one PKR above the due-date projection's minimum need."""
    horizon = HORIZON_DAYS
    dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon, freq="D")
    net = np.zeros(horizon, dtype=float)
    outstanding = invoices.loc[pd.to_datetime(invoices["paid_date"]) > as_of].copy()
    due = pd.to_datetime(outstanding["due_date"])
    paper_payment_dates = due.where(due > as_of, as_of + pd.Timedelta(days=1))
    for amount, payment_date in zip(outstanding["amount_pkr"].to_numpy(dtype=float), paper_payment_dates):
        offset = (pd.Timestamp(payment_date).normalize() - dates[0]).days
        if 0 <= offset < horizon:
            net[offset] += amount
    future_expenses = _future_expense_rows(expenses, as_of, horizon)
    for row in future_expenses.itertuples():
        offset = (pd.Timestamp(row.due_date).normalize() - dates[0]).days
        if 0 <= offset < horizon:
            net[offset] -= float(row.amount_pkr)
    minimum_without_opening = float(np.min(np.cumsum(net)))
    return max(0.0, -minimum_without_opening) + 1.0


def write_ayesha_dataset(output_dir: str | Path = OUTPUT_DIR) -> Dataset:
    """Write the scripted dataset and its fictional-scenario README note."""
    dataset = make_ayesha_dataset()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset.clients.to_csv(destination / "clients.csv", index=False)
    dataset.invoices.to_csv(destination / "invoices.csv", index=False)
    dataset.expenses.to_csv(destination / "expenses.csv", index=False)
    settings = {
        "as_of_date": dataset.settings.as_of_date,
        "opening_cash_pkr": dataset.settings.opening_cash_pkr,
        "min_buffer_pkr": dataset.settings.min_buffer_pkr,
    }
    (destination / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (destination / "README.md").write_text(
        "# Ayesha Studio scripted scenario\n\n"
        "This is a scripted demo dataset. All business names, client names, invoice amounts, dates, "
        "payment behaviors, and expense values are fictional. They are scenario inputs, not Pakistani "
        "statistics or claims about real businesses.\n",
        encoding="utf-8",
    )
    return dataset


def _simulator_is_working() -> bool:
    """Check whether the repository cash simulator is more than its stub."""
    try:
        return "STUB" not in inspect.getsource(simulate_cash).upper()
    except (OSError, TypeError):
        return False


def check_ayesha(output_dir: str | Path = OUTPUT_DIR) -> None:
    """Print the paper projection, model ranges, and ten-day chase scenario."""
    destination = Path(output_dir)
    dataset = make_ayesha_dataset()
    write_ayesha_dataset(destination)
    model, X_open, age_open, open_invoices = build_pipeline(dataset, dataset.settings.as_of_date)
    samples = model.sample(
        X_open,
        age_open,
        n=N_SIMS,
        rng=np.random.default_rng(SEED),
    )
    due_dates = pd.to_datetime(open_invoices["due_date"])
    as_of = pd.Timestamp(dataset.settings.as_of_date)
    due_cash_dates = due_dates.where(due_dates > as_of, as_of + pd.Timedelta(days=1))
    due_dates_index, due_cash = _cash_path(dataset, due_cash_dates.to_numpy())
    min_index = int(np.argmin(due_cash))
    print(
        f"Ayesha Studio scripted demo, as_of={as_of.date()}: due-date projection minimum "
        f"Rs {due_cash[min_index]:,.0f} on {due_dates_index[min_index].date()} "
        f"(opening cash Rs {dataset.settings.opening_cash_pkr:,.0f})."
    )
    event_dates = {
        pd.Timestamp(value).normalize()
        for value in due_cash_dates
        if as_of < pd.Timestamp(value) <= as_of + pd.Timedelta(days=HORIZON_DAYS)
    }
    event_dates.update(
        pd.to_datetime(_future_expense_rows(dataset.expenses, as_of, HORIZON_DAYS)["due_date"]).dt.normalize()
    )
    print("Due-date projection at cash-flow dates:")
    for event_date in sorted(event_dates):
        offset = (event_date - due_dates_index[0]).days
        print(f"  {event_date.date()}: Rs {due_cash[offset]:,.0f}")
    if _simulator_is_working():
        sim = simulate_cash(
            open_invoices,
            samples,
            _future_expense_rows(dataset.expenses, as_of, HORIZON_DAYS),
            dataset.settings.opening_cash_pkr,
            horizon=HORIZON_DAYS,
            seed=SEED,
        )
        largest = int(open_invoices["amount_pkr"].to_numpy().argmax())
        chased_samples = samples.copy()
        chased_samples[:, largest] = np.maximum(chased_samples[:, largest] - 10.0, age_open[largest] + 1.0)
        chased = simulate_cash(
            open_invoices,
            chased_samples,
            _future_expense_rows(dataset.expenses, as_of, HORIZON_DAYS),
            dataset.settings.opening_cash_pkr,
            horizon=HORIZON_DAYS,
            seed=SEED,
        )
        print(f"Model P(shortfall): {sim.p_shortfall:.3f}; chased biggest invoice 10 days earlier: {chased.p_shortfall:.3f}.")
    else:
        issue_dates = pd.to_datetime(open_invoices["issue_date"])
        low = np.quantile(samples, 0.1, axis=0)
        high = np.quantile(samples, 0.9, axis=0)
        chased_index = int(open_invoices["amount_pkr"].to_numpy().argmax())
        for index, row in enumerate(open_invoices.itertuples()):
            q10_date = issue_dates.iloc[index] + pd.Timedelta(days=float(low[index]))
            q90_date = issue_dates.iloc[index] + pd.Timedelta(days=float(high[index]))
            if index == chased_index:
                chase_start = q10_date - pd.Timedelta(days=10)
                chase_end = q90_date - pd.Timedelta(days=10)
                chase_note = f"; after 10-day chase {chase_start.date()}–{chase_end.date()}"
            else:
                chase_note = ""
            print(
                f"{row.invoice_id} ({row.client_id}, Rs {row.amount_pkr:,.0f}): "
                f"predicted pay-date P10–P90 {q10_date.date()}–{q90_date.date()}{chase_note}"
            )
        print("P(shortfall) is unavailable because src.simulate.simulate_cash is a stub; the chase output is a pay-date range only.")


def main() -> None:
    """Write the fixed scripted dataset and print its model check."""
    check_ayesha()


if __name__ == "__main__":
    main()
