import numpy as np

class DelayModel:
    """STUB (Dev A): fit() does nothing, sample() returns dummy durations."""
    def fit(self, X, T, E):
        return self

    def sample(self, X, s_obs, n, rng=None):
        """Return durations in days, shape (n, n_invoices), each > s_obs."""
        rng = rng or np.random.default_rng(42)
        base = rng.lognormal(mean=3.0, sigma=0.5, size=(n, len(X)))
        return np.maximum(base, np.asarray(s_obs) + 1)