from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Settings:
    as_of_date: str            # "YYYY-MM-DD"
    opening_cash_pkr: float
    min_buffer_pkr: float = 0.0

@dataclass
class Dataset:
    clients: pd.DataFrame      # client_id, name, segment
    invoices: pd.DataFrame     # invoice_id, client_id, issue_date, due_date, amount_pkr, paid_date
    expenses: pd.DataFrame     # expense_id, category, amount_pkr, due_date, recurrence
    settings: Settings

@dataclass
class SimResult:
    paths: np.ndarray               # (n_sims, horizon) cash by day
    p10: np.ndarray                 # (horizon,)
    p50: np.ndarray
    p90: np.ndarray
    p_short_by_day: np.ndarray      # (horizon,) P(cash < buffer)
    p_shortfall: float              # P(shortfall anywhere in horizon)
    first_short_day_dist: np.ndarray
    min_cash_dist: np.ndarray       # (n_sims,)

@dataclass
class ChaseRow:
    invoice_id: str
    client: str
    amount_pkr: float
    p_before: float
    p_after: float
    gain: float                     # p_before - p_after