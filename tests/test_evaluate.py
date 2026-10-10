"""Tests for the time-based pay-date backtest outputs."""

import numpy as np
import pandas as pd

from src.evaluate import backtest


def test_tiny_backtest_writes_outputs_with_finite_pay_date_metrics(tmp_path) -> None:
    result = backtest(
        seeds=(0, 1),
        as_of_dates=("2025-12-31",),
        modes=("easy", "hard"),
        n_clients=12,
        months=24,
        n_draws=1000,
        output_dir=tmp_path,
    )

    expected_files = (
        "pay_date_metrics.csv",
        "pay_date_summary.csv",
        "run_status.csv",
        "shortfall_status.csv",
    )
    assert all((tmp_path / name).is_file() for name in expected_files)
    metrics = pd.read_csv(tmp_path / "pay_date_metrics.csv")
    assert not metrics.empty
    assert set(metrics["mode"]) == {"easy", "hard"}
    assert np.isfinite(metrics["mae_days"].to_numpy(dtype=float)).all()
    model_rows = metrics.loc[metrics["method"] == "model"]
    assert np.isfinite(model_rows["coverage_p10_p90"].to_numpy(dtype=float)).all()
    assert set(metrics["method"]) == {"model", "due_date", "client_average", "client_bootstrap"}
    assert result["shortfall_status"].iloc[0]["status"] == "skipped"
    assert "opening cash" not in metrics.columns
