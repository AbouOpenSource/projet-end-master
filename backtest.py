"""Mini-backtest reproductible d'un portefeuille multifactoriel.

Le calcul financier reste entièrement déterministe. La couche agentique pourra
consommer le snapshot JSON produit par ce script pour rédiger une note de décision.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

FACTOR_TICKERS = {
    "Value": "VLUE",
    "Momentum": "MTUM",
    "Quality": "QUAL",
    "Low volatility": "USMV",
}
BENCHMARK = "SPY"
TICKERS = list(FACTOR_TICKERS.values()) + [BENCHMARK]
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"  # borne supérieure exclusive : dernières années complètes
OUT_OF_SAMPLE_START = "2021-01-01"
TRANSACTION_COSTS = [0.0, 0.001, 0.003]  # 0, 10 et 30 points de base
MONTHS_PER_YEAR = 12
TARGET_WEIGHTS = pd.Series(0.25, index=list(FACTOR_TICKERS.values()), dtype=float)


def prepare_directories() -> None:
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def download_prices() -> tuple[pd.DataFrame, dict]:
    """Télécharge les cours ajustés et produit un rapport de qualité."""
    raw = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )

    if raw.empty:
        raise RuntimeError("Aucune donnée téléchargée. Vérifier la connexion et les tickers.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("La réponse Yahoo Finance ne contient pas de colonne Close.")
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [TICKERS[0]]

    prices = prices.reindex(columns=TICKERS)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()

    quality = {
        "source": "Yahoo Finance via yfinance",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": TICKERS,
        "start_requested": START_DATE,
        "end_requested_exclusive": END_DATE,
        "first_observation": prices.index.min().date().isoformat(),
        "last_observation": prices.index.max().date().isoformat(),
        "rows": int(len(prices)),
        "duplicate_dates": int(prices.index.duplicated().sum()),
        "missing_values_by_ticker": {
            ticker: int(prices[ticker].isna().sum()) for ticker in TICKERS
        },
        "non_positive_values_by_ticker": {
            ticker: int((prices[ticker] <= 0).sum()) for ticker in TICKERS
        },
    }

    gaps = prices.index.to_series().diff().dt.days
    quality["maximum_calendar_gap_days"] = int(gaps.max()) if gaps.notna().any() else 0
    quality["calendar_gaps_over_7_days"] = [
        date.date().isoformat()
        for date, gap in gaps.items()
        if pd.notna(gap) and gap > 7
    ]
    large_daily_returns = (prices.pct_change().abs() > 0.25).sum()
    quality["daily_return_flags_over_25pct"] = {
        ticker: int(large_daily_returns[ticker]) for ticker in TICKERS
    }
    quality["bias_controls"] = {
        "corporate_actions": "cours auto-ajustes par yfinance ; verification institutionnelle encore necessaire",
        "look_ahead_bias": "rendements mensuels calcules apres la cloture de la periode ; aucun parametre ajuste sur les rendements futurs",
        "survivorship_bias": "non elimine dans ce prototype ETF ; un univers historique point-in-time est requis pour une etude titres",
        "liquidity_and_slippage": "non modelises separement ; les scenarios 10 et 30 pb sont des proxys de couts tout compris",
    }

    if quality["duplicate_dates"]:
        raise RuntimeError("Dates dupliquées dans les données téléchargées.")

    # Les ETF retenus ont normalement tous un historique complet sur la période.
    # On élimine les dates incomplètes plutôt que d'inventer des prix.
    prices = prices.dropna(how="any")
    if prices.empty:
        raise RuntimeError("Aucune date complète après suppression des valeurs manquantes.")

    quality["rows_after_complete_case"] = int(len(prices))

    return prices, quality


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    monthly_prices = prices.resample("ME").last()
    returns = monthly_prices.pct_change().dropna(how="any")
    monthly_prices.to_csv(DATA_DIR / "monthly_prices.csv", index_label="date")
    return returns


def simulate_factor_portfolio(
    returns: pd.DataFrame, transaction_cost: float
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Simule un rééquilibrage mensuel à poids cibles égaux."""
    factor_returns = returns[list(FACTOR_TICKERS.values())].copy()
    weights = TARGET_WEIGHTS.copy()
    net_returns: list[float] = []
    gross_returns: list[float] = []
    costs: list[float] = []
    turnovers: list[float] = []
    records: list[dict] = []

    for date, row in factor_returns.iterrows():
        gross = float((weights * (1.0 + row)).sum())
        post_return_weights = weights * (1.0 + row) / gross
        # Turnover conventionnel « one-way » : valeur des achats (ou des ventes),
        # soit la moitié du turnover brut aller-retour.
        turnover = float(0.5 * (TARGET_WEIGHTS - post_return_weights).abs().sum())
        cost = transaction_cost * turnover
        net = gross - 1.0 - cost

        gross_returns.append(gross - 1.0)
        net_returns.append(net)
        costs.append(cost)
        turnovers.append(turnover)
        records.append(
            {
                "date": date,
                "gross_return": gross - 1.0,
                "transaction_cost": cost,
                "net_return": net,
                "turnover": turnover,
                **{f"weight_{ticker}": float(value) for ticker, value in post_return_weights.items()},
            }
        )
        weights = TARGET_WEIGHTS.copy()

    diagnostics = pd.DataFrame(records).set_index("date")
    return (
        pd.Series(net_returns, index=factor_returns.index, name="Multifactor net"),
        pd.Series(gross_returns, index=factor_returns.index, name="Multifactor gross"),
        diagnostics,
    )


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def metrics(returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    if returns.empty:
        raise ValueError("Impossible de calculer les métriques sans rendement.")

    periods = len(returns)
    wealth = float((1.0 + returns).prod())
    annualized_return = wealth ** (MONTHS_PER_YEAR / periods) - 1.0
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))
        if returns.std(ddof=1) > 0
        else float("nan")
    )
    result = {
        "annualized_return": float(annualized_return),
        "annualized_volatility": annualized_volatility,
        "sharpe_zero_rf": sharpe,
        "max_drawdown": max_drawdown(returns),
        "total_return": wealth - 1.0,
        "final_value_from_100": 100.0 * wealth,
    }
    if turnover is not None:
        result["average_monthly_turnover"] = float(turnover.mean())
        result["annualized_turnover"] = float(turnover.sum() * MONTHS_PER_YEAR / periods)
    return result


def evaluation_periods(returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Retourne l echantillon complet et une periode hors echantillon fixe."""
    out_of_sample = returns.loc[returns.index >= pd.Timestamp(OUT_OF_SAMPLE_START)]
    if out_of_sample.empty:
        raise RuntimeError("La periode hors echantillon ne contient aucune observation.")
    return {
        "full_sample": returns,
        "out_of_sample_2021_2025": out_of_sample,
    }


def save_figures(wealth: pd.DataFrame, drawdowns: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    ax = wealth.plot(figsize=(11, 6), linewidth=1.8)
    ax.set_title("Évolution de 100 unités investies")
    ax.set_xlabel("")
    ax.set_ylabel("Valeur du portefeuille")
    ax.legend(loc="upper left")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cumulative_wealth.png", dpi=180)
    plt.close(fig)

    ax = drawdowns.plot(figsize=(11, 6), linewidth=1.5)
    ax.set_title("Drawdown")
    ax.set_xlabel("")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "drawdown.png", dpi=180)
    plt.close(fig)


def main() -> None:
    prepare_directories()
    prices, quality = download_prices()
    prices.to_csv(DATA_DIR / "daily_prices.csv", index_label="date")
    with (DATA_DIR / "quality_report.json").open("w", encoding="utf-8") as stream:
        json.dump(quality, stream, ensure_ascii=False, indent=2)

    returns = monthly_returns(prices)
    returns.to_csv(RESULTS_DIR / "monthly_asset_returns.csv", index_label="date")

    all_metrics: list[dict] = []
    wealth = {}
    drawdowns = {}
    snapshot: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "factor_tickers": FACTOR_TICKERS,
            "benchmark": BENCHMARK,
            "start_date": START_DATE,
            "end_date_exclusive": END_DATE,
            "rebalance": "monthly",
            "target_weights": TARGET_WEIGHTS.to_dict(),
            "transaction_costs": TRANSACTION_COSTS,
            "risk_free_rate": 0.0,
            "turnover_definition": "one_way_half_sum_absolute_weight_changes",
            "annualization": "monthly_periods_per_year_12",
        },
        "last_observation": returns.index[-1].date().isoformat(),
    }

    benchmark_returns = returns[BENCHMARK].rename("Benchmark SPY")
    wealth["Benchmark SPY"] = 100.0 * (1.0 + benchmark_returns).cumprod()
    drawdowns["Benchmark SPY"] = wealth["Benchmark SPY"] / wealth["Benchmark SPY"].cummax() - 1.0
    benchmark_metrics = metrics(benchmark_returns)
    all_metrics.append({"strategy": "Benchmark SPY", "transaction_cost": 0.0, **benchmark_metrics})

    for cost in TRANSACTION_COSTS:
        net, gross, diagnostics = simulate_factor_portfolio(returns, cost)
        label = f"Multifactor net ({cost * 10000:.0f} bps)"
        wealth[label] = 100.0 * (1.0 + net).cumprod()
        drawdowns[label] = wealth[label] / wealth[label].cummax() - 1.0
        diagnostics.to_csv(RESULTS_DIR / f"rebalance_diagnostics_{int(cost * 10000)}bps.csv")
        all_metrics.append(
            {
                "strategy": "Multifactor",
                "transaction_cost": cost,
                **metrics(net, diagnostics["turnover"]),
            }
        )

        if abs(cost - 0.001) < 1e-12:
            snapshot["last_rebalance"] = {
                "date": diagnostics.index[-1].date().isoformat(),
                "factor_returns_last_month": {
                    ticker: float(returns.loc[diagnostics.index[-1], ticker])
                    for ticker in FACTOR_TICKERS.values()
                },
                "target_weights": TARGET_WEIGHTS.to_dict(),
                "turnover": float(diagnostics.iloc[-1]["turnover"]),
                "transaction_cost": float(diagnostics.iloc[-1]["transaction_cost"]),
                "net_return": float(diagnostics.iloc[-1]["net_return"]),
            }

    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    period_rows: list[dict] = []
    for period_name, period_returns in evaluation_periods(returns).items():
        benchmark_period = period_returns[BENCHMARK].rename("Benchmark SPY")
        period_rows.append({
            "period": period_name,
            "strategy": "Benchmark SPY",
            "transaction_cost": 0.0,
            **metrics(benchmark_period),
        })
        for cost in TRANSACTION_COSTS:
            net, _, diagnostics = simulate_factor_portfolio(period_returns, cost)
            period_rows.append({
                "period": period_name,
                "strategy": "Multifactor",
                "transaction_cost": cost,
                **metrics(net, diagnostics["turnover"]),
            })
    period_metrics_frame = pd.DataFrame(period_rows)
    period_metrics_frame.to_csv(RESULTS_DIR / "period_metrics.csv", index=False)
    wealth_frame = pd.DataFrame(wealth)
    drawdown_frame = pd.DataFrame(drawdowns)
    wealth_frame.to_csv(RESULTS_DIR / "wealth.csv", index_label="date")
    drawdown_frame.to_csv(RESULTS_DIR / "drawdowns.csv", index_label="date")
    save_figures(wealth_frame, drawdown_frame)

    snapshot["metrics"] = json.loads(metrics_frame.to_json(orient="records"))
    snapshot["period_metrics"] = json.loads(period_metrics_frame.to_json(orient="records"))
    snapshot["data_quality"] = quality
    snapshot["software"] = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "yfinance": yf.__version__,
    }
    with (RESULTS_DIR / "decision_snapshot.json").open("w", encoding="utf-8") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, indent=2, default=str)

    print("Backtest terminé.")
    print(f"Période : {returns.index[0].date()} -> {returns.index[-1].date()}")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Résultats : {RESULTS_DIR}")
    print(f"Figures   : {FIGURES_DIR}")


if __name__ == "__main__":
    main()
