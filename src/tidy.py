import pandas as pd


def tidy_expenses(expenses):
    """Collapse repeated recurring rows (same category, amount and day). -> (clean, notes)"""
    if expenses is None or len(expenses) == 0 or "recurrence" not in expenses.columns:
        return expenses, []
    df = expenses.copy()
    due = pd.to_datetime(df["due_date"], errors="coerce")
    rec = df["recurrence"].astype(str).str.lower()
    slot = pd.Series(float("nan"), index=df.index)
    slot[rec == "monthly"] = due[rec == "monthly"].dt.day
    slot[rec == "weekly"] = due[rec == "weekly"].dt.dayofweek
    recurring = rec.isin(["monthly", "weekly"]) & due.notna()
    work = df.assign(_due=due, _rec=rec, _slot=slot)[recurring]
    keep = work.sort_values("_due").drop_duplicates(
        ["category", "_rec", "amount_pkr", "_slot"], keep="first").index
    drop = work.index.difference(keep)
    if len(drop) == 0:
        return expenses, []
    note = (f"Your expenses list the same recurring cost several times ({len(drop)} extra rows). "
            "I counted each recurring cost once, because the forecast repeats it automatically.")
    return df.drop(index=drop).reset_index(drop=True), [note]