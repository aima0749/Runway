import numpy as np
import pandas as pd
from src.schemas import SimResult
from config import HORIZON_DAYS, SEED

def simulate_cash(open_inv, samples, expenses, opening_cash, as_of, s_obs,
                  horizon=HORIZON_DAYS, seed=SEED, buffer=0.0,
                  weekly_sales=0.0) -> SimResult:
    """Simulate end-of-day cash paths from invoice and expense schedules."""
    samples = np.asarray(samples)
    s_obs = np.asarray(s_obs)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n_sims, n_open)")
    if samples.shape[1] != len(open_inv) or s_obs.shape != (len(open_inv),):
        raise ValueError("samples, open_inv, and s_obs must have matching invoice counts")

    n_sims, n_open = samples.shape
    inflow = np.zeros((n_sims, horizon), dtype=float)
    if n_open and horizon > 0:
        pay_days = np.maximum(1, np.ceil(samples - s_obs[None, :])).astype(int)
        valid = (pay_days >= 1) & (pay_days <= horizon)
        sim_indices, invoice_indices = np.nonzero(valid)
        day_indices = pay_days[sim_indices, invoice_indices] - 1
        amounts = open_inv["amount_pkr"].to_numpy(dtype=float)[invoice_indices]
        np.add.at(inflow, (sim_indices, day_indices), amounts)

    if weekly_sales > 0:
        inflow[:, 6::7] += weekly_sales

    as_of = pd.Timestamp(as_of)
    outflow = np.zeros(horizon, dtype=float)
    for row in expenses.itertuples(index=False):
        due_date = pd.Timestamp(row.due_date)
        amount = float(row.amount_pkr)
        recurrence = str(row.recurrence).lower()

        if recurrence == "once":
            offset = (due_date - as_of).days
            if 1 <= offset <= horizon:
                outflow[offset - 1] += amount
        elif recurrence == "weekly":
            if due_date <= as_of:
                weeks = (as_of - due_date).days // 7 + 1
                due_date += pd.Timedelta(days=7 * weeks)
            offset = (due_date - as_of).days
            while 1 <= offset <= horizon:
                outflow[offset - 1] += amount
                offset += 7
        elif recurrence == "monthly":
            month_delta = (as_of.year - due_date.year) * 12 + as_of.month - due_date.month
            occurrence_number = max(0, month_delta)
            occurrence = due_date + pd.DateOffset(months=occurrence_number)
            while occurrence <= as_of:
                occurrence_number += 1
                occurrence = due_date + pd.DateOffset(months=occurrence_number)
            while True:
                offset = (occurrence - as_of).days
                if offset > horizon:
                    break
                if offset >= 1:
                    outflow[offset - 1] += amount
                occurrence_number += 1
                occurrence = due_date + pd.DateOffset(months=occurrence_number)
        else:
            raise ValueError(f"unsupported expense recurrence: {row.recurrence}")

    paths = opening_cash + np.cumsum(inflow - outflow[None, :], axis=1)
    short = paths < buffer
    first = np.where(short.any(axis=1), short.argmax(axis=1) + 1, -1)
    return SimResult(
        paths=paths,
        p10=np.percentile(paths, 10, axis=0),
        p50=np.percentile(paths, 50, axis=0),
        p90=np.percentile(paths, 90, axis=0),
        p_short_by_day=short.mean(axis=0),
        p_shortfall=float(short.any(axis=1).mean()),
        first_short_day_dist=first,
        min_cash_dist=paths.min(axis=1),
    )