"""CSV templates and the upload wrapper around Dev A's validator."""
import csv
import io

import pandas as pd

TEMPLATES = {
    "clients.csv": (["client_id", "name", "segment"],
                    [["C1", "Example Traders", "sme"]]),
    "invoices.csv": (["invoice_id", "client_id", "issue_date", "due_date", "amount_pkr", "paid_date"],
                     [["INV1", "C1", "2025-01-05", "2025-02-04", "100000", "2025-02-10"],
                      ["INV2", "C1", "2025-02-01", "2025-03-03", "150000", ""]]),
    "expenses.csv": (["expense_id", "category", "amount_pkr", "due_date", "recurrence"],
                     [["E1", "rent", "80000", "2025-03-01", "monthly"],
                      ["E2", "salaries", "250000", "2025-02-20", "monthly"]]),
}
TEMPLATE_NAMES = list(TEMPLATES)
# Notes: segment = corporate | sme | individual. Leave paid_date blank for unpaid invoices.
# recurrence = once | monthly | weekly. ONE row per recurring expense, not one row per month.


def template_csv(name: str) -> str:
    header, rows = TEMPLATES[name]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def validate_upload(clients_file, invoices_file, expenses_file, settings):
    """-> (Dataset or None, list of plain-language errors).
    NOTE: adjust the load_and_validate call below if Dev A's real signature differs."""
    from src.data import load_and_validate
    try:
        files = {"clients": pd.read_csv(clients_file),
                 "invoices": pd.read_csv(invoices_file),
                 "expenses": pd.read_csv(expenses_file)}
    except Exception as ex:
        return None, [f"I could not read one of the files as a CSV: {ex}"]
    try:
        result = load_and_validate(files, settings)
    except Exception as ex:
        return None, [f"The files could not be validated: {ex}"]
    if isinstance(result, tuple):
        ds, errors = result[0], list(result[1] or [])
    else:
        ds, errors = result, []
    if errors or ds is None or not hasattr(ds, "settings"):
        return None, errors or ["The files could not be validated."]
    return ds, []