import numpy as np
import pandas as pd
from src.schemas import Dataset, Settings

def generate_business(seed=42, n_clients=25, months=24, mode="easy") -> Dataset:
    """STUB (Dev A): tiny dummy dataset with the right columns."""
    clients = pd.DataFrame({
        "client_id": ["C1", "C2"],
        "name": ["Client One", "Client Two"],
        "segment": ["corporate", "sme"],
    })
    invoices = pd.DataFrame({
        "invoice_id": ["INV1", "INV2", "INV3"],
        "client_id": ["C1", "C2", "C1"],
        "issue_date": ["2026-08-01", "2026-08-10", "2026-09-01"],
        "due_date": ["2026-08-31", "2026-09-09", "2026-10-01"],
        "amount_pkr": [200000.0, 150000.0, 300000.0],
        "paid_date": ["2026-09-15", "2026-09-20", None],
    })
    expenses = pd.DataFrame({
        "expense_id": ["E1", "E2"],
        "category": ["rent", "salaries"],
        "amount_pkr": [80000.0, 250000.0],
        "due_date": ["2026-10-01", "2026-10-20"],
        "recurrence": ["monthly", "monthly"],
    })
    return Dataset(clients, invoices, expenses,
                   Settings(as_of_date="2026-10-10", opening_cash_pkr=300000.0))