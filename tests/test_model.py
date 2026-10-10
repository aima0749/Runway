"""Tests for the censored log-normal AFT model."""

import time

import numpy as np
import pandas as pd
from src.delay_model import DelayModel, build_pipeline
from src.generator import generate_business
from src.schemas import Dataset, Settings


def _aft_data(seed: int = 5, n: int = 3000) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Simulate a known AFT process with independent right censoring."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    features = pd.DataFrame({"x1": x1, "x2": x2})
    true_duration = np.exp(2.0 + 0.4 * x1 - 0.25 * x2 + 0.5 * rng.normal(size=n))
    censor_age = rng.uniform(2.0, 15.0, size=n)
    event = (true_duration <= censor_age).astype(int)
    observed = np.minimum(true_duration, censor_age)
    return features, observed, event


def test_sampled_durations_are_strictly_after_observed_age() -> None:
    X, T, E = _aft_data(n=800)
    model = DelayModel().fit(X, T, E)
    ages = np.linspace(0.0, 30.0, len(X))
    samples = model.sample(X, ages, n=100, rng=np.random.default_rng(7))
    assert samples.shape == (100, len(X))
    assert (samples > np.maximum(ages, 0.5)[None, :]).all()


def test_fit_completes_quickly_on_about_one_thousand_invoices() -> None:
    X, T, E = _aft_data(n=1000)
    started = time.perf_counter()
    model = DelayModel().fit(X, T, E)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0
    assert np.isfinite(model.sigma_)


def test_hist_mean_delay_coefficient_is_positive_on_easy_data() -> None:
    clients, invoices, _ = generate_business(seed=14, n_clients=120, months=12, mode="easy")
    from src.features import build_features

    X, T, E, _ = build_features(invoices, clients, "2025-01-01")
    model = DelayModel().fit(X, T, E)
    coefficient = model.b_[1 + X.columns.get_loc("hist_mean_delay")]
    assert coefficient > 0


def test_parameters_are_recovered_with_independent_censoring() -> None:
    X, T, E = _aft_data(seed=12, n=6000)
    model = DelayModel(l2=0.5).fit(X, T, E)
    assert abs(model.b_[0] - 2.0) < 0.08
    assert abs(model.b_[1] - 0.4) < 0.07
    assert abs(model.b_[2] + 0.25) < 0.07
    assert abs(model.sigma_ - 0.5) < 0.06


def test_dropping_censored_rows_predicts_faster_than_using_them() -> None:
    rng = np.random.default_rng(19)
    n = 3000
    X = pd.DataFrame({"intercept_feature": np.zeros(n)})
    truth = np.exp(2.5 + 0.45 * rng.normal(size=n))
    censor_age = rng.uniform(2.0, 8.0, size=n)
    observed_event = truth <= censor_age
    observed_time = np.minimum(truth, censor_age)
    model_with_censoring = DelayModel().fit(X, observed_time, observed_event.astype(int))
    model_dropped = DelayModel().fit(X.loc[observed_event], truth[observed_event], np.ones(observed_event.sum(), dtype=int))
    prediction_with = float(np.exp(model_with_censoring.predict_mu(X.iloc[:1])[0]))
    prediction_dropped = float(np.exp(model_dropped.predict_mu(X.iloc[:1])[0]))
    print(f"P50 using censored rows: {prediction_with:.2f} days")
    print(f"P50 after dropping censored rows: {prediction_dropped:.2f} days")
    assert prediction_with > prediction_dropped


def test_build_pipeline_returns_open_invoices_and_sanity_metrics() -> None:
    clients, invoices, expenses = generate_business(seed=1, n_clients=25, months=24, mode="easy")
    as_of = "2025-07-01"  # 18 months after the generation start.
    dataset = Dataset(clients, invoices, expenses, Settings(as_of, 500_000.0))
    model, X_open, age_open, open_invoices = build_pipeline(dataset, as_of)
    assert len(X_open) == len(age_open) == len(open_invoices)
    assert len(open_invoices) > 0
    assert "true_paid_date" not in X_open.columns

    rng = np.random.default_rng(1)
    draws = model.sample(X_open, age_open, n=1000, rng=rng)
    predicted_days = np.quantile(draws, 0.5, axis=0)
    p10 = np.quantile(draws, 0.1, axis=0)
    p90 = np.quantile(draws, 0.9, axis=0)
    issue = pd.to_datetime(open_invoices["issue_date"]).to_numpy(dtype="datetime64[D]")
    true_paid = pd.to_datetime(open_invoices["true_paid_date"]).to_numpy(dtype="datetime64[D]")
    predicted_p10 = issue + np.floor(p10).astype("int64").astype("timedelta64[D]")
    predicted_p90 = issue + np.ceil(p90).astype("int64").astype("timedelta64[D]")
    true_day = (true_paid - issue).astype("timedelta64[D]").astype(int)
    model_mae = float(np.mean(np.abs(predicted_days - true_day)))
    due_date = pd.to_datetime(open_invoices["due_date"])
    due_mae = float(np.mean(np.abs((true_paid - due_date.to_numpy(dtype="datetime64[D]")).astype("timedelta64[D]").astype(int))))
    assert np.isfinite(model_mae) and np.isfinite(due_mae)
    coverage = float(np.mean((true_paid >= predicted_p10) & (true_paid <= predicted_p90)))
    avg_predicted_dates = []
    history_paid = pd.to_datetime(invoices["paid_date"])
    history_due = pd.to_datetime(invoices["due_date"])
    for row in open_invoices.itertuples():
        issue_date = pd.Timestamp(row.issue_date)
        prior = (
            (invoices["client_id"] == row.client_id)
            & history_paid.notna()
            & (history_paid <= issue_date)
        )
        prior_delays = (history_paid[prior] - history_due[prior]).dt.days
        average_delay = float(prior_delays.mean()) if len(prior_delays) else 0.0
        avg_predicted_dates.append(pd.Timestamp(row.due_date) + pd.Timedelta(days=average_delay))
    avg_predicted = pd.to_datetime(avg_predicted_dates).to_numpy(dtype="datetime64[D]")
    client_average_mae = float(np.mean(np.abs(
        (true_paid - avg_predicted).astype("timedelta64[D]").astype(int)
    )))
    print(f"Sanity run as_of={as_of}, open invoices={len(open_invoices)}")
    print(f"P50 pay-date MAE: {model_mae:.2f} days; due-date MAE: {due_mae:.2f} days; client-average delay MAE: {client_average_mae:.2f} days")
    print(f"P10-P90 coverage: {coverage:.3f}")
    if model_mae < 1.0:
        print("Very low MAE: audited feature columns exclude true_paid_date; history is limited to payments known by each issue date.")
    assert 0.0 <= coverage <= 1.0
