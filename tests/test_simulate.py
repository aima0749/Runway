import numpy as np
import pandas as pd

from src.simulate import simulate_cash


AS_OF = pd.Timestamp("2026-10-10")


def test_same_seed_gives_identical_output():
    invoices = pd.DataFrame({"amount_pkr": [50000.0]})
    expenses = pd.DataFrame({
        "amount_pkr": [20000.0],
        "due_date": ["2026-10-20"],
        "recurrence": ["once"],
    })
    args = (invoices, np.array([[25.0], [30.0]]), expenses, 100000.0, AS_OF, np.array([10]))

    first = simulate_cash(*args, horizon=60, seed=7, weekly_sales=1000.0)
    second = simulate_cash(*args, horizon=60, seed=7, weekly_sales=1000.0)

    for field in (
        "paths", "p10", "p50", "p90", "p_short_by_day",
        "first_short_day_dist", "min_cash_dist",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert first.p_shortfall == second.p_shortfall


def test_worked_invoice_and_recurring_expense_example():
    invoices = pd.DataFrame({
        "issue_date": ["2026-09-30"],
        "amount_pkr": [50000.0],
    })
    expenses = pd.DataFrame({
        "amount_pkr": [20000.0, 5000.0],
        "due_date": ["2026-10-20", "2026-10-15"],
        "recurrence": ["once", "monthly"],
    })

    result = simulate_cash(
        invoices,
        np.array([[25.0]]),
        expenses,
        100000.0,
        AS_OF,
        np.array([10]),
        horizon=60,
    )

    np.testing.assert_array_equal(
        result.paths[0, [3, 4, 9, 14, 35]],
        [100000.0, 95000.0, 75000.0, 125000.0, 120000.0],
    )


def test_sample_just_above_observed_age_pays_on_day_one():
    invoices = pd.DataFrame({"amount_pkr": [100.0]})

    result = simulate_cash(
        invoices,
        np.array([[10.01]]),
        pd.DataFrame(columns=["amount_pkr", "due_date", "recurrence"]),
        0.0,
        AS_OF,
        np.array([10]),
        horizon=5,
    )

    np.testing.assert_array_equal(result.paths[0], [100.0] * 5)


def test_once_expense_due_in_past_is_ignored():
    expenses = pd.DataFrame({
        "amount_pkr": [1000.0],
        "due_date": ["2026-10-09"],
        "recurrence": ["once"],
    })

    result = simulate_cash(
        pd.DataFrame(columns=["amount_pkr"]),
        np.empty((2, 0)),
        expenses,
        100.0,
        AS_OF,
        np.empty(0, dtype=int),
        horizon=5,
    )

    np.testing.assert_array_equal(result.paths, np.full((2, 5), 100.0))


def test_large_expense_causes_shortfall_in_every_simulation():
    expenses = pd.DataFrame({
        "amount_pkr": [1000000.0],
        "due_date": ["2026-10-11"],
        "recurrence": ["once"],
    })

    result = simulate_cash(
        pd.DataFrame(columns=["amount_pkr"]),
        np.empty((100, 0)),
        expenses,
        0.0,
        AS_OF,
        np.empty(0, dtype=int),
        horizon=60,
    )

    assert result.p_shortfall == 1.0


def test_cash_percentiles_are_ordered_every_day():
    invoices = pd.DataFrame({"amount_pkr": [100.0, 250.0]})
    samples = np.tile(np.array([[2.0, 7.0]]), (1000, 1))

    result = simulate_cash(
        invoices,
        samples,
        pd.DataFrame(columns=["amount_pkr", "due_date", "recurrence"]),
        50.0,
        AS_OF,
        np.array([1, 1]),
        horizon=60,
    )

    assert np.all(result.p10 <= result.p50)
    assert np.all(result.p50 <= result.p90)