import numpy as np
from src.schemas import SimResult
from config import HORIZON_DAYS, N_SIMS, SEED

def simulate_cash(open_inv, samples, expenses, opening_cash,
                  horizon=HORIZON_DAYS, seed=SEED) -> SimResult:
    """STUB (Dev B): dummy result with correct shapes."""
    rng = np.random.default_rng(seed)
    paths = opening_cash + np.cumsum(rng.normal(-5000, 20000, (N_SIMS, horizon)), axis=1)
    return SimResult(
        paths=paths,
        p10=np.percentile(paths, 10, axis=0),
        p50=np.percentile(paths, 50, axis=0),
        p90=np.percentile(paths, 90, axis=0),
        p_short_by_day=(paths < 0).mean(axis=0),
        p_shortfall=float((paths < 0).any(axis=1).mean()),
        first_short_day_dist=np.full(N_SIMS, 20),
        min_cash_dist=paths.min(axis=1),
    )