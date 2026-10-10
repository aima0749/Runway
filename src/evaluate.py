"""Time-based synthetic backtests for pay-date prediction."""

# duration T = days from issue_date to paid_date (model target, always >= 1).
# delay = days from due_date to paid_date (history feature and baseline).
# An unpaid invoice is censored: T = as_of - issue_date, E = 0. A paid invoice
# has E = 1. Never drop unpaid invoices.

from __future__ import annotations

import inspect
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import HORIZON_DAYS, SEED
from src.delay_model import DelayModel
from src.features import build_features
from src.generator import censor_at, generate_business
from src.simulate import simulate_cash

DEFAULT_AS_OF_DATES = (
    "2024-06-30",
    "2024-12-31",
    "2025-06-30",
    "2025-12-31",
)
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "eval" / "results"
ENABLE_SHORTFALL_EVALUATION = True


def _simulator_is_working() -> bool:
    """Return false while simulate_cash still contains its repository stub."""
    try:
        source = inspect.getsource(simulate_cash)
    except (OSError, TypeError):
        return False
    return "STUB" not in source.upper()


def _client_history_delays(
    observed: pd.DataFrame,
    client_id: str,
    issue_date: pd.Timestamp,
) -> np.ndarray:
    """Return known same-client delays paid by the invoice issue date."""
    paid = pd.to_datetime(observed["paid_date"], errors="coerce")
    due = pd.to_datetime(observed["due_date"], errors="raise")
    mask = (
        (observed["client_id"] == client_id)
        & paid.notna()
        & (paid <= issue_date)
    )
    return (paid.loc[mask] - due.loc[mask]).dt.days.to_numpy(dtype=float)


def _fold_pay_date_metrics(
    seed: int,
    mode: str,
    as_of: str,
    train_as_of: pd.Timestamp,
    clients: pd.DataFrame,
    invoices: pd.DataFrame,
    n_draws: int,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object] | None]:
    """Fit one temporal fold and score its open invoices."""
    train_observed = censor_at(invoices, train_as_of.date())
    X_train, T_train, E_train, _ = build_features(train_observed, clients, train_as_of)
    if len(X_train) == 0:
        return [], {"seed": seed, "mode": mode, "as_of_date": as_of, "status": "skipped", "reason": "empty training window"}, None
    model = DelayModel().fit(X_train, T_train, E_train)

    cutoff = pd.Timestamp(as_of)
    observed = censor_at(invoices, cutoff.date())
    X_all, _, _, aligned = build_features(observed, clients, cutoff)
    issue_dates = pd.to_datetime(aligned["issue_date"])
    test_mask = (issue_dates > train_as_of) & aligned["paid_date"].isna()
    open_invoices = aligned.loc[test_mask].copy()
    X_open = X_all.loc[test_mask].copy()
    true_paid = pd.to_datetime(open_invoices["true_paid_date"], errors="coerce")
    has_truth = true_paid.notna().to_numpy()
    open_invoices = open_invoices.loc[has_truth].copy()
    X_open = X_open.loc[has_truth].copy()
    true_paid = true_paid.loc[has_truth]
    if open_invoices.empty:
        return [], {"seed": seed, "mode": mode, "as_of_date": as_of, "status": "skipped", "reason": "no open invoices with known true pay dates"}, None

    issue_dates = pd.to_datetime(open_invoices["issue_date"])
    due_dates = pd.to_datetime(open_invoices["due_date"])
    age = (cutoff - issue_dates).dt.days.to_numpy(dtype=float)
    fold_seed = int(seed * 1_000_003 + cutoff.toordinal())
    model_draws = model.sample(
        X_open,
        age,
        n=n_draws,
        rng=np.random.default_rng(fold_seed),
    )
    if model_draws.shape != (n_draws, len(open_invoices)):
        raise ValueError("DelayModel.sample returned an unexpected shape")

    true_duration = (true_paid - issue_dates).dt.days.to_numpy(dtype=float)
    model_median = np.quantile(model_draws, 0.5, axis=0)
    model_p10 = np.quantile(model_draws, 0.1, axis=0)
    model_p90 = np.quantile(model_draws, 0.9, axis=0)
    due_prediction = (due_dates - issue_dates).dt.days.to_numpy(dtype=float)
    due_cash_duration = np.maximum(due_prediction, age + 1.0)
    average_predictions: list[float] = []
    bootstrap_predictions: list[float] = []
    baseline_rng = np.random.default_rng(fold_seed + 1)
    for position, row in enumerate(open_invoices.itertuples()):
        row_issue = pd.Timestamp(row.issue_date)
        history = _client_history_delays(observed, row.client_id, row_issue)
        average_delay = float(history.mean()) if len(history) else 0.0
        average_predictions.append(float((pd.Timestamp(row.due_date) - row_issue).days + average_delay))
        if len(history):
            draws = baseline_rng.choice(history, size=1000, replace=True)
            bootstrap_delay = float(np.median(draws))
        else:
            bootstrap_delay = 0.0
        bootstrap_predictions.append(float((pd.Timestamp(row.due_date) - row_issue).days + bootstrap_delay))

    cold_start = X_open["n_prior"].to_numpy(dtype=float) < 3
    methods: dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray | None]] = {
        "model": (model_median, model_p10, model_p90),
        "due_date": (due_prediction, None, None),
        "client_average": (np.asarray(average_predictions), None, None),
        "client_bootstrap": (np.asarray(bootstrap_predictions), None, None),
    }
    rows: list[dict[str, object]] = []
    for cohort_name, cohort_mask in (("cold_start", cold_start), ("not_cold_start", ~cold_start)):
        if not np.any(cohort_mask):
            continue
        for method, (point, low, high) in methods.items():
            errors = np.abs(point[cohort_mask] - true_duration[cohort_mask])
            coverage = (
                float(np.mean((true_duration[cohort_mask] >= low[cohort_mask]) & (true_duration[cohort_mask] <= high[cohort_mask])))
                if low is not None and high is not None
                else np.nan
            )
            rows.append({
                "seed": seed,
                "mode": mode,
                "as_of_date": as_of,
                "train_as_of": train_as_of.date().isoformat(),
                "cohort": cohort_name,
                "method": method,
                "n_invoices": int(cohort_mask.sum()),
                "mae_days": float(np.mean(errors)),
                "coverage_p10_p90": coverage,
            })

    context = {
        "model_draws": model_draws,
        "due_durations": due_cash_duration,
        "average_durations": np.asarray(average_predictions),
        "age": age,
        "open_invoices": open_invoices,
        "as_of": cutoff,
    }
    return rows, {
        "seed": seed,
        "mode": mode,
        "as_of_date": as_of,
        "train_as_of": train_as_of.date().isoformat(),
        "status": "completed",
        "n_open_test_invoices": len(open_invoices),
    }, context


def _write_skipped_shortfall_status(path: Path, reason: str) -> pd.DataFrame:
    """Save the skip status when cash simulation is unavailable."""
    status = pd.DataFrame([{"status": "skipped", "reason": reason}])
    status.to_csv(path, index=False)
    return status


def _shortfall_reliability(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> pd.DataFrame:
    """Summarize predicted and observed shortfall rates in five fixed bins."""
    edges = np.linspace(0.0, 1.0, 6)
    labels = np.clip(np.digitize(probabilities, edges[1:-1], right=True), 0, 4)
    rows = []
    for index in range(5):
        mask = labels == index
        rows.append({
            "bin": index + 1,
            "p_min": float(edges[index]),
            "p_max": float(edges[index + 1]),
            "n": int(mask.sum()),
            "mean_predicted": float(probabilities[mask].mean()) if np.any(mask) else np.nan,
            "observed_rate": float(outcomes[mask].mean()) if np.any(mask) else np.nan,
        })
    return pd.DataFrame(rows)


def _future_expenses(expenses: pd.DataFrame, as_of: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Expand monthly synthetic expenses through the cash forecast horizon."""
    end = as_of + pd.Timedelta(days=horizon)
    rows: list[dict[str, object]] = []
    monthly = expenses.loc[expenses["recurrence"] == "monthly"].copy()
    if monthly.empty:
        return pd.DataFrame(columns=["expense_id", "category", "amount_pkr", "due_date", "recurrence"])
    monthly["_due"] = pd.to_datetime(monthly["due_date"])
    amounts = monthly.groupby("category")["amount_pkr"].mean().to_dict()
    days = monthly.groupby("category")["_due"].apply(lambda values: int(round(values.dt.day.median()))).to_dict()
    index = 0
    cursor = date(as_of.year, as_of.month, 1)
    while pd.Timestamp(cursor) <= end:
        for category, amount in amounts.items():
            day = int(days[category])
            due = date(cursor.year, cursor.month, min(day, monthrange(cursor.year, cursor.month)[1]))
            if as_of.date() < due <= end.date():
                index += 1
                rows.append({
                    "expense_id": f"F{index:05d}",
                    "category": category,
                    "amount_pkr": float(amount),
                    "due_date": due.isoformat(),
                    "recurrence": "monthly",
                })
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    return pd.DataFrame(rows, columns=["expense_id", "category", "amount_pkr", "due_date", "recurrence"])


def _cash_minimum(
    opening_cash: float,
    invoices: pd.DataFrame,
    payment_dates: np.ndarray,
    expenses: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int,
) -> float:
    """Return the minimum daily cash from specified invoice payment dates."""
    flows = np.zeros(horizon, dtype=float)
    first_day = as_of.date() + timedelta(days=1)
    for amount, value in zip(invoices["amount_pkr"].to_numpy(dtype=float), payment_dates):
        payment_day = pd.Timestamp(value).date()
        offset = (payment_day - first_day).days
        if 0 <= offset < horizon:
            flows[offset] += amount
    for row in expenses.itertuples():
        due_day = pd.Timestamp(row.due_date).date()
        offset = (due_day - first_day).days
        if 0 <= offset < horizon:
            flows[offset] -= float(row.amount_pkr)
    return float(np.min(opening_cash + np.cumsum(flows))) if horizon else opening_cash


def _opening_cash_for_paper_projection(
    invoices: pd.DataFrame,
    due_dates: np.ndarray,
    expenses: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int,
) -> float:
    """Find the smallest opening cash that leaves due-date cash positive."""
    paper_minimum = _cash_minimum(0.0, invoices, due_dates, expenses, as_of, horizon)
    cushion = float(np.spacing(max(abs(paper_minimum), 1.0)))
    return max(0.0, -paper_minimum) + cushion


def _historical_shortfall_rate(
    invoices: pd.DataFrame,
    expenses: pd.DataFrame,
    before: pd.Timestamp,
    horizon: int,
) -> float | None:
    """Estimate a constant base rate from prior rolling cash windows only."""
    issue = pd.to_datetime(invoices["issue_date"])
    paid = pd.to_datetime(invoices["paid_date"], errors="coerce")
    first = issue.min().normalize() + pd.Timedelta(days=horizon)
    window_end = first
    outcomes: list[float] = []
    while window_end <= before:
        window_start = window_end - pd.Timedelta(days=horizon)
        open_mask = (issue <= window_start) & (paid.isna() | (paid > window_start))
        open_invoices = invoices.loc[open_mask].copy()
        future_expenses = _future_expenses(expenses, window_start, horizon)
        if len(open_invoices):
            due = pd.to_datetime(open_invoices["due_date"])
            due_projection = np.maximum(
                (due - pd.to_datetime(open_invoices["issue_date"])).dt.days.to_numpy(dtype=float),
                (window_start - pd.to_datetime(open_invoices["issue_date"])).dt.days.to_numpy(dtype=float) + 1,
            )
            paper_dates = (pd.to_datetime(open_invoices["issue_date"]) + pd.to_timedelta(due_projection, unit="D")).to_numpy()
            paper_dates = np.maximum(paper_dates, (window_start + pd.Timedelta(days=1)).to_datetime64())
            opening = _opening_cash_for_paper_projection(open_invoices, paper_dates, future_expenses, window_start, horizon)
            truth = pd.to_datetime(open_invoices["paid_date"], errors="coerce").fillna(pd.Timestamp.max)
            outcomes.append(float(_cash_minimum(opening, open_invoices, truth.to_numpy(), future_expenses, window_start, horizon) < 0))
        window_end += pd.Timedelta(days=30)
    return float(np.mean(outcomes)) if outcomes else None


def _run_shortfall_fold(
    context: dict[str, object],
    expenses: pd.DataFrame,
    invoices: pd.DataFrame,
    train_as_of: pd.Timestamp,
    seed: int,
    mode: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Score model and deterministic baselines with the cash simulator."""
    open_invoices = context["open_invoices"]
    model_draws = np.asarray(context["model_draws"], dtype=float)
    due_durations = np.asarray(context["due_durations"], dtype=float)
    average_durations = np.asarray(context["average_durations"], dtype=float)
    age = np.asarray(context["age"], dtype=float)
    as_of = pd.Timestamp(context["as_of"])
    horizon = HORIZON_DAYS
    future_expenses = _future_expenses(expenses, as_of, horizon)
    issue_dates = pd.to_datetime(open_invoices["issue_date"])
    due_dates = pd.to_datetime(open_invoices["due_date"])
    due_cash_dates = np.maximum(due_dates.to_numpy(dtype="datetime64[ns]"), (as_of + pd.Timedelta(days=1)).to_datetime64())
    opening_cash = _opening_cash_for_paper_projection(open_invoices, due_cash_dates, future_expenses, as_of, horizon)

    true_dates = pd.to_datetime(open_invoices["true_paid_date"], errors="coerce")
    actual_outcome = float(_cash_minimum(opening_cash, open_invoices, true_dates.to_numpy(), future_expenses, as_of, horizon) < 0)
    due_samples = np.repeat(due_durations[None, :], model_draws.shape[0], axis=0)
    average_samples = np.repeat(np.maximum(average_durations, age + 1.0)[None, :], model_draws.shape[0], axis=0)
    simulator_results = {
        "model": simulate_cash(open_invoices, model_draws, future_expenses, opening_cash, horizon=horizon, seed=SEED + seed),
        "due_date": simulate_cash(open_invoices, due_samples, future_expenses, opening_cash, horizon=horizon, seed=SEED + seed),
        "client_average": simulate_cash(open_invoices, average_samples, future_expenses, opening_cash, horizon=horizon, seed=SEED + seed),
    }

    base_rate = _historical_shortfall_rate(invoices, expenses, train_as_of, horizon)
    if base_rate is None:
        raise ValueError("no historical windows are available for the constant base-rate baseline")
    probabilities = {
        "model": float(simulator_results["model"].p_shortfall),
        "due_date": float(simulator_results["due_date"].p_shortfall),
        "client_average": float(simulator_results["client_average"].p_shortfall),
        "constant_base_rate": base_rate,
    }
    rows = []
    for method, probability in probabilities.items():
        rows.append({
            "seed": seed,
            "mode": mode,
            "as_of_date": as_of.date().isoformat(),
            "n_open_invoices": len(open_invoices),
            "opening_cash_pkr": opening_cash,
            "observed_shortfall": actual_outcome,
            "method": method,
            "predicted_probability": probability,
            "brier_score": float((probability - actual_outcome) ** 2),
        })
    reliability_rows = []
    for method, probability in probabilities.items():
        bins = _shortfall_reliability(
            np.asarray([probability], dtype=float),
            np.asarray([actual_outcome], dtype=float),
        )
        for row in bins.to_dict("records"):
            reliability_rows.append({"seed": seed, "mode": mode, "method": method, **row})
    return rows, reliability_rows


def backtest(
    seeds: Iterable[int] = range(SEED, SEED + 25),
    as_of_dates: Iterable[str] = DEFAULT_AS_OF_DATES,
    modes: Iterable[str] = ("easy", "hard"),
    n_clients: int = 25,
    months: int = 24,
    n_draws: int = 1000,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
    include_shortfall: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """Run deterministic, chronological pay-date evaluation and save CSVs."""
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    seed_values = tuple(int(value) for value in seeds)
    dates = tuple(pd.Timestamp(value).date().isoformat() for value in as_of_dates)
    mode_values = tuple(modes)
    if any(mode not in {"easy", "hard"} for mode in mode_values):
        raise ValueError("modes may contain only 'easy' and 'hard'")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    shortfall_contexts: list[tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.Timestamp, int, str]] = []
    requested_shortfall = ENABLE_SHORTFALL_EVALUATION if include_shortfall is None else include_shortfall
    simulator_working = _simulator_is_working()

    # Process dates in order so any future gated baseline can use past folds only.
    ordered_dates = tuple(sorted(dates))
    for mode in mode_values:
        for seed in seed_values:
            clients, invoices, expenses = generate_business(
                seed=seed,
                n_clients=n_clients,
                months=months,
                mode=mode,
            )
            for as_of in ordered_dates:
                cutoff = pd.Timestamp(as_of)
                train_as_of = cutoff - pd.DateOffset(months=3)
                fold_rows, fold_status, context = _fold_pay_date_metrics(
                    seed=seed,
                    mode=mode,
                    as_of=as_of,
                    train_as_of=train_as_of,
                    clients=clients,
                    invoices=invoices,
                    n_draws=n_draws,
                )
                metrics_rows.extend(fold_rows)
                status_rows.append(fold_status)
                if requested_shortfall and simulator_working and context is not None:
                    shortfall_contexts.append((context, expenses, invoices, train_as_of, seed, mode))

    metric_columns = [
        "seed", "mode", "as_of_date", "train_as_of", "cohort", "method",
        "n_invoices", "mae_days", "coverage_p10_p90",
    ]
    metrics = pd.DataFrame(metrics_rows, columns=metric_columns)
    status = pd.DataFrame(status_rows)
    summary = (
        metrics.groupby(["mode", "cohort", "method"], as_index=False)
        .agg(
            folds=("mae_days", "size"),
            n_invoices=("n_invoices", "sum"),
            mae_days=("mae_days", "mean"),
            coverage_p10_p90=("coverage_p10_p90", "mean"),
        )
        if not metrics.empty
        else pd.DataFrame(columns=["mode", "cohort", "method", "folds", "n_invoices", "mae_days", "coverage_p10_p90"])
    )
    metrics_path = destination / "pay_date_metrics.csv"
    summary_path = destination / "pay_date_summary.csv"
    status_path = destination / "run_status.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    status.to_csv(status_path, index=False)

    if not requested_shortfall:
        shortfall_status = _write_skipped_shortfall_status(destination / "shortfall_status.csv", "disabled by evaluation flag")
    elif not simulator_working:
        shortfall_status = _write_skipped_shortfall_status(destination / "shortfall_status.csv", "src.simulate.simulate_cash is a stub")
    else:
        shortfall_rows: list[dict[str, object]] = []
        reliability_rows: list[dict[str, object]] = []
        failures: list[str] = []
        for context, expense_rows, invoice_rows, train_as_of, seed, mode in shortfall_contexts:
            try:
                fold_shortfall, fold_reliability = _run_shortfall_fold(
                    context, expense_rows, invoice_rows, train_as_of, seed, mode,
                )
                shortfall_rows.extend(fold_shortfall)
                reliability_rows.extend(fold_reliability)
            except (ValueError, TypeError, AttributeError) as exc:
                failures.append(str(exc))
        shortfall_metrics = pd.DataFrame(shortfall_rows)
        reliability = pd.DataFrame(reliability_rows)
        if not shortfall_metrics.empty:
            shortfall_metrics.to_csv(destination / "shortfall_metrics.csv", index=False)
            reliability.to_csv(destination / "shortfall_reliability.csv", index=False)
            try:
                import plotly.graph_objects as go

                fig = go.Figure()
                for method, group in shortfall_metrics.groupby("method"):
                    binned = _shortfall_reliability(
                        group["predicted_probability"].to_numpy(dtype=float),
                        group["observed_shortfall"].to_numpy(dtype=float),
                    )
                    visible = binned[binned["n"] > 0]
                    fig.add_trace(go.Scatter(
                        x=visible["mean_predicted"],
                        y=visible["observed_rate"],
                        mode="lines+markers",
                        name=method,
                    ))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="ideal", line={"dash": "dash"}))
                fig.update_layout(
                    title="Shortfall probability reliability",
                    xaxis_title="Mean predicted probability",
                    yaxis_title="Observed shortfall rate",
                    xaxis={"range": [0, 1]},
                    yaxis={"range": [0, 1]},
                )
                fig.write_image(str(destination / "shortfall_reliability.png"))
            except (ImportError, ValueError, RuntimeError) as exc:
                failures.append(f"reliability PNG unavailable: {exc}")
        if shortfall_metrics.empty or failures:
            reason = "; ".join(failures) if failures else "no eligible shortfall folds"
            shortfall_status = _write_skipped_shortfall_status(destination / "shortfall_status.csv", reason)
        else:
            shortfall_status = pd.DataFrame([{"status": "completed", "reason": ""}])
            shortfall_status.to_csv(destination / "shortfall_status.csv", index=False)

    completed = int((status.get("status", pd.Series(dtype=str)) == "completed").sum())
    total_rows = int(metrics["n_invoices"].sum()) if not metrics.empty else 0
    print(
        f"Pay-date backtest: {completed} completed temporal folds, {total_rows} invoice-cohort observations; "
        f"modes={', '.join(mode_values)}. Training uses the prior three months' cutoff and test invoices issued after it. "
        "Shortfall opening cash is designed so the on-paper due-date projection barely stays positive; "
        f"shortfall status: {shortfall_status.iloc[0]['status']} ({shortfall_status.iloc[0]['reason']})."
    )
    if not summary.empty:
        print(summary.to_string(index=False))
    if requested_shortfall and simulator_working and "shortfall_metrics" in locals() and not shortfall_metrics.empty:
        shortfall_summary = shortfall_metrics.groupby(["mode", "method"], as_index=False)["brier_score"].mean()
        print(shortfall_summary.to_string(index=False))
    results = {
        "pay_date_metrics": metrics,
        "pay_date_summary": summary,
        "run_status": status,
        "shortfall_status": shortfall_status,
    }
    if requested_shortfall and simulator_working and "shortfall_metrics" in locals():
        results["shortfall_metrics"] = shortfall_metrics
        results["shortfall_reliability"] = reliability
    return results


if __name__ == "__main__":
    backtest()
