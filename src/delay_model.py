"""Right-censored log-normal accelerated failure time model."""

# duration T = days from issue_date to paid_date (model target, always >= 1).
# delay = days from due_date to paid_date (used for history features).
# An unpaid invoice is censored: T = as_of - issue_date, E = 0. A paid invoice
# has E = 1. Never drop unpaid invoices.

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from src.features import build_features
from src.generator import censor_at
from src.schemas import Dataset


class DelayModel:
    """Fit a regularized log-normal AFT model with right censoring."""

    def __init__(self, l2: float = 0.5) -> None:
        """Create a model with the requested L2 penalty strength."""
        if l2 < 0:
            raise ValueError("l2 must be nonnegative")
        self.l2 = float(l2)
        self.feature_names_: list[str] | None = None
        self.impute_means_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.b_: np.ndarray | None = None
        self.sigma_: float | None = None
        self._standardized_b: np.ndarray | None = None

    def _raw_matrix(self, X: pd.DataFrame | np.ndarray, fitting: bool = False) -> np.ndarray:
        """Convert features to a numeric array in the fitted column order."""
        if isinstance(X, pd.DataFrame):
            if fitting:
                self.feature_names_ = [str(column) for column in X.columns]
                values = X.to_numpy(dtype=float)
            else:
                if self.feature_names_ is None:
                    raise RuntimeError("fit must be called before prediction")
                missing = set(self.feature_names_).difference(X.columns)
                if missing:
                    raise ValueError(f"prediction features missing columns: {sorted(missing)}")
                values = X.loc[:, self.feature_names_].to_numpy(dtype=float)
        else:
            values = np.asarray(X, dtype=float)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if not fitting and self.feature_names_ is not None and values.shape[1] != len(self.feature_names_):
                raise ValueError("prediction feature count differs from fitted feature count")
        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        return values

    def _transform(self, X: pd.DataFrame | np.ndarray, fitting: bool = False) -> np.ndarray:
        """Impute using training means and standardize features."""
        values = self._raw_matrix(X, fitting=fitting)
        if fitting:
            finite = np.isfinite(values)
            counts = finite.sum(axis=0)
            means = np.divide(
                np.where(finite, values, 0.0).sum(axis=0),
                counts,
                out=np.zeros(values.shape[1], dtype=float),
                where=counts > 0,
            )
            self.impute_means_ = means
            filled = np.where(np.isfinite(values), values, means)
            self.mean_ = filled.mean(axis=0)
            std = filled.std(axis=0)
            self.std_ = np.where(std > 0.0, std, 1.0)
        elif self.impute_means_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("fit must be called before prediction")
        else:
            filled = np.where(np.isfinite(values), values, self.impute_means_)
        assert self.mean_ is not None and self.std_ is not None
        return (filled - self.mean_) / self.std_

    def fit(self, X: pd.DataFrame | np.ndarray, T: np.ndarray, E: np.ndarray) -> DelayModel:
        """Fit AFT coefficients by maximizing the censored likelihood."""
        transformed = self._transform(X, fitting=True)
        if len(transformed) == 0:
            raise ValueError("cannot fit on an empty dataset")
        durations = np.asarray(T, dtype=float).reshape(-1)
        events_raw = np.asarray(E, dtype=float).reshape(-1)
        if len(durations) != len(transformed) or len(events_raw) != len(transformed):
            raise ValueError("X, T, and E must have matching row counts")
        if np.any(~np.isfinite(durations)) or np.any(durations <= 0):
            raise ValueError("T must contain finite positive durations")
        if np.any(~np.isfinite(events_raw)) or np.any(~np.isin(events_raw, [0.0, 1.0])):
            raise ValueError("E must contain only 0 or 1")
        events = events_raw.astype(int)

        log_t = np.log(durations)
        design = np.column_stack((np.ones(len(transformed)), transformed))
        n_features = design.shape[1]

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            coefficients = parameters[:n_features]
            log_sigma = float(parameters[-1])
            sigma = float(np.exp(log_sigma))
            mu = design @ coefficients
            residual = (log_t - mu) / sigma
            log_likelihood = np.empty(len(log_t), dtype=float)
            gradient_mu = np.empty(len(log_t), dtype=float)
            gradient_log_sigma = np.empty(len(log_t), dtype=float)

            observed = events == 1
            censored = ~observed
            log_likelihood[observed] = -log_sigma - 0.5 * residual[observed] ** 2
            gradient_mu[observed] = residual[observed] / sigma
            gradient_log_sigma[observed] = -1.0 + residual[observed] ** 2

            if np.any(censored):
                r_censored = residual[censored]
                log_survival = norm.logsf(r_censored)
                log_likelihood[censored] = log_survival
                mills = np.exp(np.minimum(norm.logpdf(r_censored) - log_survival, 700.0))
                gradient_mu[censored] = mills / sigma
                gradient_log_sigma[censored] = mills * r_censored

            penalty = 0.5 * self.l2 * float(np.dot(coefficients[1:], coefficients[1:]))
            loss = -float(log_likelihood.sum()) + penalty
            gradient_beta = -(design.T @ gradient_mu)
            gradient_beta[1:] += self.l2 * coefficients[1:]
            gradient_sigma = -float(gradient_log_sigma.sum())
            gradient = np.concatenate((gradient_beta, [gradient_sigma]))
            return loss, gradient

        initial = np.zeros(n_features + 1, dtype=float)
        initial[0] = float(np.mean(log_t))
        initial[-1] = np.log(max(float(np.std(log_t)), 0.3))
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(None, None)] * n_features + [(-5.0, 3.0)],
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not result.success and not np.isfinite(result.fun):
            raise RuntimeError(f"AFT optimization failed: {result.message}")

        self._standardized_b = result.x[:n_features].copy()
        self.sigma_ = float(np.exp(result.x[-1]))
        assert self.mean_ is not None and self.std_ is not None
        raw_slopes = self._standardized_b[1:] / self.std_
        raw_intercept = self._standardized_b[0] - float(np.dot(raw_slopes, self.mean_))
        self.b_ = np.concatenate(([raw_intercept], raw_slopes))
        return self

    def predict_mu(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict the linear mean of log duration for each invoice."""
        if self._standardized_b is None:
            raise RuntimeError("fit must be called before prediction")
        transformed = self._transform(X)
        design = np.column_stack((np.ones(len(transformed)), transformed))
        return design @ self._standardized_b

    def sample(
        self,
        X: pd.DataFrame | np.ndarray,
        s_obs: np.ndarray | float,
        n: int,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """Draw durations conditional on remaining unpaid at observed age."""
        if self.sigma_ is None:
            raise RuntimeError("fit must be called before sampling")
        if n < 1:
            raise ValueError("n must be positive")
        mu = self.predict_mu(X)
        ages = np.asarray(s_obs, dtype=float)
        if ages.ndim == 0:
            ages = np.full(len(mu), float(ages))
        ages = np.broadcast_to(ages.reshape(-1), (len(mu),))
        ages = np.maximum(ages, 0.5)
        random = np.random.default_rng(rng) if isinstance(rng, (int, type(None))) else rng
        assert isinstance(random, np.random.Generator)

        threshold = (np.log(ages) - mu) / self.sigma_
        survival = norm.sf(threshold)
        uniform = random.uniform(size=(n, len(mu)))
        quantile = np.clip(1.0 - uniform * survival[None, :], 1e-9, 1.0 - 1e-9)
        durations = np.exp(mu[None, :] + self.sigma_ * norm.ppf(quantile))
        durations = np.minimum(durations, 365.0)
        durations = np.maximum(durations, np.nextafter(ages[None, :], np.inf))
        return durations


def build_pipeline(
    dataset: Dataset,
    as_of: str | pd.Timestamp,
    l2: float = 0.5,
) -> tuple[DelayModel, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Censor a dataset, fit the model, and return open invoice predictions."""
    observed = censor_at(dataset.invoices, as_of)
    X, T, E, aligned = build_features(observed, dataset.clients, as_of)
    model = DelayModel(l2=l2).fit(X, T, E)
    open_mask = aligned["paid_date"].isna().to_numpy()
    X_open = X.loc[open_mask].copy()
    open_invoices = aligned.loc[open_mask].copy()
    age_open = (
        pd.Timestamp(as_of) - pd.to_datetime(open_invoices["issue_date"])
    ).dt.days.to_numpy(dtype=float)
    return model, X_open, age_open, open_invoices
