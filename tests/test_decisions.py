from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from config import AMBER_MAX, CHASE_SHIFT_DAYS, GREEN_MAX
from src.decisions import rank_chase, run_what_if, status
from src.schemas import Settings
from src.simulate import simulate_cash


AS_OF = pd.Timestamp("2026-10-10")
EXPENSE_COLUMNS = ["expense_id", "category", "amount_pkr", "due_date", "recurrence"]


def make_engine(amounts, pay_day, opening_cash=100.0, expenses=None, n_sims=100):
    clients = pd.DataFrame({
        "client_id": [f"C{i}" for i in range(len(amounts))],
        "name": [f"Client {i}" for i in range(len(amounts))],
    })
    open_inv = pd.DataFrame({
        "invoice_id": [f"INV{i}" for i in range(len(amounts))],
        "client_id": clients["client_id"],
        "amount_pkr": amounts,
    })
    samples = np.full((n_sims, len(amounts)), float(pay_day))
    s_obs = np.zeros(len(amounts), dtype=int)
    if expenses is None:
        expenses = pd.DataFrame(columns=EXPENSE_COLUMNS)
    ds = SimpleNamespace(
        clients=clients,
        expenses=expenses,
        settings=Settings(AS_OF.strftime("%Y-%m-%d"), opening_cash),
    )
    sim = simulate_cash(open_inv, samples, expenses, opening_cash, AS_OF, s_obs)
    return SimpleNamespace(
        ds=ds,
        open_inv=open_inv,
        samples=samples,
        s_obs=s_obs,
        as_of=AS_OF,
        sim=sim,
    )


def test_status_uses_configured_thresholds():
    assert status(GREEN_MAX - 0.01) == "green"
    assert status(GREEN_MAX) == "amber"
    assert status(AMBER_MAX) == "amber"
    assert status(AMBER_MAX + 0.01) == "red"


def test_rank_chase_gains_are_not_negative():
    expense_day = CHASE_SHIFT_DAYS + 11
    expenses = pd.DataFrame([{
        "expense_id": "E1",
        "category": "rent",
        "amount_pkr": 150.0,
        "due_date": AS_OF + pd.DateOffset(days=expense_day),
        "recurrence": "once",
    }], columns=EXPENSE_COLUMNS)
    engine = make_engine([100.0, 50.0, 20.0], CHASE_SHIFT_DAYS + 20,
                         opening_cash=100.0, expenses=expenses)

    rows = rank_chase(engine)

    assert all(row.gain >= -1e-12 for row in rows)


def test_rank_chase_is_sorted_and_respects_top_n():
    engine = make_engine([100.0, 50.0, 20.0], CHASE_SHIFT_DAYS + 20)

    rows = rank_chase(engine, top_n=2)

    assert len(rows) == 2
    assert [row.gain for row in rows] == sorted(
        [row.gain for row in rows], reverse=True
    )


def test_bigger_invoice_has_no_smaller_gain_in_hand_made_case():
    expense_day = CHASE_SHIFT_DAYS + 11
    expenses = pd.DataFrame([{
        "expense_id": "E1",
        "category": "rent",
        "amount_pkr": 150.0,
        "due_date": AS_OF + pd.DateOffset(days=expense_day),
        "recurrence": "once",
    }], columns=EXPENSE_COLUMNS)
    engine = make_engine([100.0, 20.0], CHASE_SHIFT_DAYS + 20,
                         opening_cash=100.0, expenses=expenses)

    rows = rank_chase(engine, top_n=2)
    gain_by_amount = {row.amount_pkr: row.gain for row in rows}

    assert gain_by_amount[100.0] >= gain_by_amount[20.0]


def test_hire_what_if_never_lowers_shortfall_probability():
    engine = make_engine([100.0], 5)

    result = run_what_if(engine, {"type": "hire", "monthly": 25.0, "start": "2026-10-15"})

    assert result["p_after"] >= result["p_before"]


def test_delay_client_never_lowers_shortfall_probability():
    expense_day = CHASE_SHIFT_DAYS + 11
    expenses = pd.DataFrame([{
        "expense_id": "E1",
        "category": "rent",
        "amount_pkr": 150.0,
        "due_date": AS_OF + pd.DateOffset(days=expense_day),
        "recurrence": "once",
    }], columns=EXPENSE_COLUMNS)
    engine = make_engine([100.0], expense_day - 1, opening_cash=100.0, expenses=expenses)

    result = run_what_if(engine, {"type": "delay_client", "client_id": "C0", "days": 20})

    assert result["p_after"] >= result["p_before"]


def test_decision_calls_do_not_modify_engine_state():
    engine = make_engine([100.0, 50.0], CHASE_SHIFT_DAYS + 20)
    samples_before = engine.samples.copy()
    invoices_before = engine.open_inv.copy(deep=True)
    expenses_before = engine.ds.expenses.copy(deep=True)
    sim_before = engine.sim
    sim_fields_before = {
        field: getattr(sim_before, field).copy()
        if isinstance(getattr(sim_before, field), np.ndarray)
        else getattr(sim_before, field)
        for field in (
            "paths", "p10", "p50", "p90", "p_short_by_day",
            "p_shortfall", "first_short_day_dist", "min_cash_dist",
        )
    }

    rank_chase(engine, top_n=1)
    run_what_if(engine, {"type": "hire", "monthly": 25.0, "start": "2026-10-15"})
    run_what_if(engine, {"type": "add_expense", "amount": 10.0, "due_date": "2026-10-20"})
    run_what_if(engine, {"type": "delay_client", "client_id": "C0", "days": 3})
    run_what_if(engine, {"type": "weekly_sales", "amount": 15.0})

    np.testing.assert_array_equal(engine.samples, samples_before)
    pdt.assert_frame_equal(engine.open_inv, invoices_before)
    pdt.assert_frame_equal(engine.ds.expenses, expenses_before)
    assert engine.sim is sim_before
    for field, expected in sim_fields_before.items():
        actual = getattr(engine.sim, field)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def test_unknown_change_type_raises_value_error():
    engine = make_engine([100.0], 5)

    with pytest.raises(ValueError, match="unknown change type"):
        run_what_if(engine, {"type": "not_a_change"})