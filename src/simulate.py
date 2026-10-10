import numpy as np
from src.schemas import SimResult
from config import HORIZON_DAYS, N_SIMS, SEED

def simulate_cash(open_inv, samples, expenses, opening_cash, as_of, s_obs,
                  horizon=HORIZON_DAYS, seed=SEED, buffer=0.0,
                  weekly_sales=0.0) -> SimResult:
    """STUB (Dev B): dummy result with correct shapes.
    Contract: paths[:, j] is cash at end of day as_of + (j+1).
    samples = durations since issue; pay offset = max(1, ceil(sample - s_obs))."""
    rng = np.random.default_rng(seed)
    paths = opening_cash + np.cumsum(rng.normal(-5000, 20000, (N_SIMS, horizon)), axis=1)
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