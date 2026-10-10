import pandas as pd
from src.tidy import tidy_expenses


def _rows(n, rec="monthly"):
    return pd.DataFrame({
        "expense_id": [f"E{i}" for i in range(n)], "category": ["rent"] * n,
        "amount_pkr": [60000.0] * n, "recurrence": [rec] * n,
        "due_date": [(pd.Timestamp("2024-01-01") + pd.DateOffset(months=i)).strftime("%Y-%m-%d")
                     for i in range(n)]})

def test_repeated_monthly_rows_collapse_to_first():
    clean, notes = tidy_expenses(_rows(12))
    assert len(clean) == 1 and clean["due_date"].iloc[0] == "2024-01-01" and notes

def test_clean_data_untouched():
    assert tidy_expenses(_rows(1))[1] == []

def test_once_rows_never_collapse():
    assert len(tidy_expenses(_rows(3, "once"))[0]) == 3

def test_different_amounts_kept():
    df = _rows(2); df.loc[1, "amount_pkr"] = 70000.0
    assert len(tidy_expenses(df)[0]) == 2

def test_engine_counts_ayesha_expenses_once():
    from src import tools
    tools.set_engine(tools.default_dataset())
    assert len(tools.get_engine().ds.expenses) <= 6
    r = tools.call_tool("get_runway")
    top = tools.call_tool("rank_chase", {"top_n": 1})["ranked"][0]
    assert top["p_before_pct"] == r["p_shortfall_pct"]