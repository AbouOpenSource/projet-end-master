from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import backtest


def one_month_returns(**overrides: float) -> pd.DataFrame:
    values = {ticker: 0.0 for ticker in backtest.FACTOR_TICKERS.values()}
    values.update(overrides)
    return pd.DataFrame([values], index=pd.DatetimeIndex(["2024-01-31"]))


def test_turnover_uses_one_way_convention_and_costs_are_proportional() -> None:
    returns = one_month_returns(VLUE=0.10)

    net_0, gross, diagnostics_0 = backtest.simulate_factor_portfolio(returns, 0.0)
    net_10, _, diagnostics_10 = backtest.simulate_factor_portfolio(returns, 0.001)

    post_weights = diagnostics_0.iloc[0][
        [f"weight_{ticker}" for ticker in backtest.FACTOR_TICKERS.values()]
    ].to_numpy(dtype=float)
    target = backtest.TARGET_WEIGHTS.to_numpy(dtype=float)
    expected_turnover = 0.5 * np.abs(target - post_weights).sum()

    assert diagnostics_0.iloc[0]["turnover"] == pytest.approx(expected_turnover)
    assert diagnostics_10.iloc[0]["turnover"] == pytest.approx(expected_turnover)
    assert net_0.iloc[0] == pytest.approx(gross.iloc[0])
    assert net_10.iloc[0] == pytest.approx(
        gross.iloc[0] - 0.001 * expected_turnover
    )


def test_metrics_use_monthly_annualization_and_monthly_sharpe() -> None:
    returns = pd.Series(
        [0.01, 0.02] * 6,
        index=pd.date_range("2024-01-31", periods=12, freq="ME"),
    )
    turnover = pd.Series(0.02, index=returns.index)

    result = backtest.metrics(returns, turnover)

    expected_wealth = float((1.0 + returns).prod())
    expected_volatility = float(returns.std(ddof=1) * np.sqrt(12))
    expected_sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(12))

    assert result["annualized_return"] == pytest.approx(expected_wealth - 1.0)
    assert result["annualized_volatility"] == pytest.approx(expected_volatility)
    assert result["sharpe_zero_rf"] == pytest.approx(expected_sharpe)
    assert result["average_monthly_turnover"] == pytest.approx(0.02)
    assert result["annualized_turnover"] == pytest.approx(0.24)


def test_metrics_reject_empty_series() -> None:
    with pytest.raises(ValueError, match="sans rendement"):
        backtest.metrics(pd.Series(dtype=float))


def _assert_metric_row(actual: pd.Series, expected: dict[str, float]) -> None:
    for key, expected_value in expected.items():
        actual_value = actual[key]
        assert math.isclose(
            float(actual_value),
            float(expected_value),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ), f"{key}: {actual_value} != {expected_value}"


def test_archived_metrics_match_recomputed_outputs() -> None:
    returns = pd.read_csv(
        backtest.RESULTS_DIR / "monthly_asset_returns.csv",
        parse_dates=["date"],
    ).set_index("date")
    archived = pd.read_csv(backtest.RESULTS_DIR / "metrics.csv")

    expected_rows: list[dict[str, object]] = [
        {
            "strategy": "Benchmark SPY",
            "transaction_cost": 0.0,
            "metrics": backtest.metrics(returns[backtest.BENCHMARK]),
        }
    ]
    for cost in backtest.TRANSACTION_COSTS:
        net, _, diagnostics = backtest.simulate_factor_portfolio(returns, cost)
        expected_rows.append(
            {
                "strategy": "Multifactor",
                "transaction_cost": cost,
                "metrics": backtest.metrics(net, diagnostics["turnover"]),
            }
        )

    for expected in expected_rows:
        mask = (
            (archived["strategy"] == expected["strategy"])
            & np.isclose(
                archived["transaction_cost"],
                float(expected["transaction_cost"]),
            )
        )
        matches = archived.loc[mask]
        assert len(matches) == 1
        _assert_metric_row(matches.iloc[0], expected["metrics"])


def test_archived_period_metrics_match_recomputed_outputs() -> None:
    returns = pd.read_csv(
        backtest.RESULTS_DIR / "monthly_asset_returns.csv",
        parse_dates=["date"],
    ).set_index("date")
    archived = pd.read_csv(backtest.RESULTS_DIR / "period_metrics.csv")

    periods = {
        "full_sample": returns,
        "out_of_sample_2021_2025": returns.loc[
            returns.index >= pd.Timestamp(backtest.OUT_OF_SAMPLE_START)
        ],
    }
    for period_name, period_returns in periods.items():
        expected_rows = [
            (
                "Benchmark SPY",
                0.0,
                backtest.metrics(period_returns[backtest.BENCHMARK]),
            )
        ]
        for cost in backtest.TRANSACTION_COSTS:
            net, _, diagnostics = backtest.simulate_factor_portfolio(period_returns, cost)
            expected_rows.append(
                (
                    "Multifactor",
                    cost,
                    backtest.metrics(net, diagnostics["turnover"]),
                )
            )

        for strategy, cost, expected_metrics in expected_rows:
            mask = (
                (archived["period"] == period_name)
                & (archived["strategy"] == strategy)
                & np.isclose(archived["transaction_cost"], cost)
            )
            matches = archived.loc[mask]
            assert len(matches) == 1
            _assert_metric_row(matches.iloc[0], expected_metrics)
