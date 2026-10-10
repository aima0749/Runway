import numpy as np
import pandas as pd

FEATURE_COLS = ["n_prior", "hist_mean_delay", "hist_std_delay",
                "hist_share_late_gt7", "last_delay", "log_amount",
                "terms", "due_day_ge_25"]

def build_features(invoices: pd.DataFrame) -> pd.DataFrame:
    """STUB (Dev A): dummy feature matrix, one row per invoice."""
    n = len(invoices)
    return pd.DataFrame({
        "n_prior": np.zeros(n), "hist_mean_delay": np.full(n, 10.0),
        "hist_std_delay": np.full(n, 5.0), "hist_share_late_gt7": np.full(n, 0.3),
        "last_delay": np.full(n, 10.0),
        "log_amount": np.log(invoices["amount_pkr"].astype(float).values),
        "terms": np.full(n, 30.0), "due_day_ge_25": np.zeros(n),
    }, index=invoices.index)