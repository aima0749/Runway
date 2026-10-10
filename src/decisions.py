from collections.abc import Mapping

import numpy as np
import pandas as pd

from config import AMBER_MAX, CHASE_SHIFT_DAYS, CHASE_TOP_N, GREEN_MAX
from src.schemas import ChaseRow
from src.simulate import simulate_cash

def status(p: float) -> str:
    if p < GREEN_MAX:
        return "green"
    if p <= AMBER_MAX:
        return "amber"
    return "red"

def _simulate(engine, samples, expenses, weekly_sales=0.0):
    return simulate_cash(
        engine.open_inv.copy(deep=True),
        np.asarray(samples).copy(),
        expenses.copy(deep=True),
        engine.ds.settings.opening_cash_pkr,
        as_of=engine.as_of,
        s_obs=np.asarray(engine.s_obs).copy(),
        buffer=engine.ds.settings.min_buffer_pkr,
        weekly_sales=weekly_sales,
    )


def rank_chase(engine, top_n=CHASE_TOP_N) -> list[ChaseRow]:
    if engine.open_inv.empty:
        return []

    limit = max(0, int(top_n))
    if limit == 0:
        return []

    amounts = engine.open_inv["amount_pkr"].to_numpy(dtype=float)
    candidate_positions = np.argsort(-amounts, kind="stable")[:CHASE_TOP_N]
    p_before = float(engine.sim.p_shortfall)
    rows = []

    for position in candidate_positions:
        shifted_samples = np.asarray(engine.samples).copy()
        shifted_samples[:, position] -= CHASE_SHIFT_DAYS
        after = _simulate(engine, shifted_samples, engine.ds.expenses)

        invoice = engine.open_inv.iloc[position]
        client_id = invoice["client_id"]
        client_names = engine.ds.clients.loc[
            engine.ds.clients["client_id"] == client_id, "name"
        ]
        client_name = str(client_names.iloc[0]) if len(client_names) else str(client_id)
        p_after = float(after.p_shortfall)
        rows.append(ChaseRow(
            invoice_id=str(invoice["invoice_id"]),
            client=client_name,
            amount_pkr=float(invoice["amount_pkr"]),
            p_before=p_before,
            p_after=p_after,
            gain=p_before - p_after,
        ))

    rows.sort(key=lambda row: row.gain, reverse=True)
    return rows[:limit]


def _required(change, key, change_type):
    if key not in change:
        raise ValueError(f"{change_type} requires '{key}'.")
    return change[key]


def _number(value, field, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"'{field}' must be a number.")
    number = float(value)
    if not np.isfinite(number) or number < minimum:
        qualifier = "greater than zero" if minimum > 0 else "zero or greater"
        raise ValueError(f"'{field}' must be {qualifier}.")
    return number


def _date(value, field):
    try:
        date = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"'{field}' must be a valid date.") from exc
    if pd.isna(date):
        raise ValueError(f"'{field}' must be a valid date.")
    return date


def _new_expense_id(expenses, prefix):
    used = set(expenses["expense_id"].dropna().astype(str))
    suffix = 1
    while f"{prefix}_{suffix}" in used:
        suffix += 1
    return f"{prefix}_{suffix}"


def _append_expense(expenses, category, amount, due_date, recurrence, prefix):
    required_columns = {"expense_id", "category", "amount_pkr", "due_date", "recurrence"}
    missing = required_columns.difference(expenses.columns)
    if missing:
        raise ValueError(f"engine expenses are missing columns: {', '.join(sorted(missing))}.")

    expense_copy = expenses.copy(deep=True)
    row = {
        "expense_id": _new_expense_id(expense_copy, prefix),
        "category": category,
        "amount_pkr": amount,
        "due_date": due_date,
        "recurrence": recurrence,
    }
    if expense_copy.empty:
        return pd.DataFrame([row], columns=expense_copy.columns)
    return pd.concat([expense_copy, pd.DataFrame([row], columns=expense_copy.columns)], ignore_index=True)


def run_what_if(engine, change) -> dict:
    if not isinstance(change, Mapping):
        raise ValueError("change must be a dictionary.")
    if "type" not in change:
        raise ValueError("change requires a 'type'.")

    change_type = change["type"]
    if not isinstance(change_type, str):
        raise ValueError("change 'type' must be a string.")
    if change_type not in {"hire", "add_expense", "delay_client", "weekly_sales"}:
        raise ValueError(f"unknown change type {change_type!r}.")

    samples = np.asarray(engine.samples).copy()
    expenses = engine.ds.expenses.copy(deep=True)
    weekly_sales = 0.0

    if change_type == "hire":
        amount = _number(_required(change, "monthly", change_type), "monthly", 0.0)
        if amount == 0:
            raise ValueError("'monthly' must be greater than zero.")
        start = _date(_required(change, "start", change_type), "start")
        expenses = _append_expense(expenses, "salaries", amount, start, "monthly", "what_if_hire")
    elif change_type == "add_expense":
        amount = _number(_required(change, "amount", change_type), "amount", 0.0)
        if amount == 0:
            raise ValueError("'amount' must be greater than zero.")
        due_date = _date(_required(change, "due_date", change_type), "due_date")
        expenses = _append_expense(expenses, "other", amount, due_date, "once", "what_if_expense")
    elif change_type == "delay_client":
        client_id = _required(change, "client_id", change_type)
        if not isinstance(client_id, str) or not client_id.strip():
            raise ValueError("'client_id' must be a non-empty string.")
        days = _number(_required(change, "days", change_type), "days", 0.0)
        known_clients = engine.ds.clients["client_id"].eq(client_id).any()
        if not known_clients:
            raise ValueError(f"client_id {client_id!r} was not found.")
        client_mask = engine.open_inv["client_id"].eq(client_id).to_numpy()
        samples[:, client_mask] += days
    else:
        weekly_sales = _number(_required(change, "amount", change_type), "amount", 0.0)

    before = engine.sim
    after = _simulate(engine, samples, expenses, weekly_sales=weekly_sales)
    return {
        "p_before": float(before.p_shortfall),
        "p_after": float(after.p_shortfall),
        "min_cash_p50_before": float(np.median(before.min_cash_dist)),
        "min_cash_p50_after": float(np.median(after.min_cash_dist)),
    }