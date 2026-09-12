"""Orchestration LangGraph de la couche agentique de demonstration.

Le graphe ne contient pas de modele de langage et ne passe aucun ordre. Il
illustre la separation entre lecture de resultats deterministes, controle de
qualite, analyse, garde de risque et validation humaine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FACTOR_TICKERS = {
    "Value": "VLUE",
    "Momentum": "MTUM",
    "Quality": "QUAL",
    "Low volatility": "USMV",
}


class WorkflowState(TypedDict, total=False):
    requested_date: str | None
    quality: dict[str, Any]
    diagnostics: pd.DataFrame
    monthly_returns: pd.DataFrame
    date: str
    rebalance: dict[str, Any]
    quality_ok: bool
    quality_flags: list[str]
    risk_status: str
    risk_alerts: list[str]
    note_markdown: str
    trace: dict[str, Any]


def load_inputs(state: WorkflowState) -> dict[str, Any]:
    with (DATA_DIR / "quality_report.json").open(encoding="utf-8") as stream:
        quality = json.load(stream)
    diagnostics = pd.read_csv(
        RESULTS_DIR / "rebalance_diagnostics_10bps.csv",
        parse_dates=["date"],
    ).set_index("date")
    monthly_returns = pd.read_csv(
        RESULTS_DIR / "monthly_asset_returns.csv",
        parse_dates=["date"],
    ).set_index("date")
    return {
        "quality": quality,
        "diagnostics": diagnostics,
        "monthly_returns": monthly_returns,
    }


def quality_gate(state: WorkflowState) -> dict[str, Any]:
    quality = state["quality"]
    flags: list[str] = []
    for key, label in (
        ("duplicate_dates", "dates dupliquees"),
        ("missing_values_by_ticker", "valeurs manquantes"),
        ("non_positive_values_by_ticker", "prix non positifs"),
        ("calendar_gaps_over_7_days", "gaps calendaires"),
        ("daily_return_flags_over_25pct", "mouvements quotidiens extrêmes"),
    ):
        value = quality.get(key, 0)
        if isinstance(value, dict):
            if any(number for number in value.values()):
                flags.append(label)
        elif value:
            flags.append(label)
    return {"quality_ok": not flags, "quality_flags": flags}


def route_quality(state: WorkflowState) -> str:
    return "rebalance" if state["quality_ok"] else "blocked"


def rebalance_analyst(state: WorkflowState) -> dict[str, Any]:
    diagnostics = state["diagnostics"]
    returns = state["monthly_returns"]
    requested = state.get("requested_date")
    date = pd.Timestamp(requested) if requested else diagnostics.index[-1]
    if date not in diagnostics.index or date not in returns.index:
        raise ValueError(f"Date de reequilibrage indisponible : {date.date()}")
    row = diagnostics.loc[date]
    factor_returns = {
        label: float(returns.loc[date, ticker])
        for label, ticker in FACTOR_TICKERS.items()
    }
    rebalance = {
        "date": date.date().isoformat(),
        "factor_returns": factor_returns,
        "net_return": float(row["net_return"]),
        "turnover": float(row["turnover"]),
        "transaction_cost": float(row["transaction_cost"]),
        "target_weights": {ticker: 0.25 for ticker in FACTOR_TICKERS.values()},
    }
    return {"date": rebalance["date"], "rebalance": rebalance}


def risk_gate(state: WorkflowState) -> dict[str, Any]:
    rebalance = state["rebalance"]
    alerts = [
        "validation humaine obligatoire",
        "aucun moteur institutionnel de limites, liquidite ou slippage n'est connecte",
        "aucune permission d'execution n'est exposee a ce workflow",
    ]
    if rebalance["turnover"] > 0.10:
        alerts.append("turnover superieur a 10 % : escalade requise")
    return {"risk_status": "a valider", "risk_alerts": alerts}


def supervisor(state: WorkflowState) -> dict[str, Any]:
    rebalance = state["rebalance"]

    def pct(value: float) -> str:
        return f"{value * 100:+.2f} %"

    lines = [
        f"## Reequilibrage du {state['date']}",
        "",
        "Le portefeuille multifactoriel reste sur ses ponderations cibles de 25 % par sleeve.",
        "La note est produite par un workflow LangGraph a partir de sorties quantitatives structurees.",
        "",
        "**Constats quantitatifs :**",
    ]
    lines.extend(
        f"- {label} : {pct(value)}"
        for label, value in rebalance["factor_returns"].items()
    )
    lines.extend(
        [
            f"- Rendement net : {pct(rebalance['net_return'])}",
            f"- Turnover : {pct(rebalance['turnover'])}",
            f"- Cout de transaction : {rebalance['transaction_cost'] * 100:.5f} %",
            "",
            f"**Avis du superviseur :** {state['risk_status']}. Aucune modification de poids ni aucun ordre ne sont generes.",
            "",
            "**Alertes :**",
        ]
    )
    lines.extend(f"- {alert}" for alert in state["risk_alerts"])
    lines.extend(
        [
            "",
            "La decision finale reste humaine et doit etre archivee avec le journal JSON associe.",
        ]
    )
    trace = {
        "workflow": "langgraph_deterministic_supervisor",
        "date": state["date"],
        "quality_ok": state["quality_ok"],
        "quality_flags": state["quality_flags"],
        "risk_status": state["risk_status"],
        "risk_alerts": state["risk_alerts"],
        "rebalance": rebalance,
        "permissions": {
            "can_change_weights": False,
            "can_send_orders": False,
            "human_validation_required": True,
        },
    }
    return {"note_markdown": "\n".join(lines) + "\n", "trace": trace}


def blocked_note(state: WorkflowState) -> dict[str, Any]:
    date = state.get("requested_date") or "date inconnue"
    note = (
        f"## Reequilibrage du {date}\n\n"
        "Le workflow est bloque par le controle qualite des donnees. Aucun avis de portefeuille, "
        "aucune modification de poids et aucun ordre ne sont generes.\n\n"
        "**Motifs :**\n"
        + "\n".join(f"- {flag}" for flag in state["quality_flags"])
        + "\n"
    )
    trace = {
        "workflow": "langgraph_deterministic_supervisor",
        "quality_ok": False,
        "quality_flags": state["quality_flags"],
        "permissions": {"can_change_weights": False, "can_send_orders": False},
    }
    return {"note_markdown": note, "trace": trace, "date": date}


def persist(state: WorkflowState) -> dict[str, Any]:
    date = state["date"]
    note_path = RESULTS_DIR / f"agent_decision_note_{date}.md"
    trace_path = RESULTS_DIR / f"agent_decision_trace_{date}.json"
    note_path.write_text(state["note_markdown"], encoding="utf-8")
    trace_path.write_text(
        json.dumps(state["trace"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {}


def build_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("load_inputs", load_inputs)
    builder.add_node("quality_gate", quality_gate)
    builder.add_node("rebalance_analyst", rebalance_analyst)
    builder.add_node("risk_gate", risk_gate)
    builder.add_node("supervisor", supervisor)
    builder.add_node("blocked_note", blocked_note)
    builder.add_node("persist", persist)
    builder.add_edge(START, "load_inputs")
    builder.add_edge("load_inputs", "quality_gate")
    builder.add_conditional_edges(
        "quality_gate",
        route_quality,
        {"rebalance": "rebalance_analyst", "blocked": "blocked_note"},
    )
    builder.add_edge("rebalance_analyst", "risk_gate")
    builder.add_edge("risk_gate", "supervisor")
    builder.add_edge("supervisor", "persist")
    builder.add_edge("blocked_note", "persist")
    builder.add_edge("persist", END)
    return builder.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        help="date mensuelle YYYY-MM-DD ; par defaut, les deux dernieres dates",
    )
    args = parser.parse_args()
    diagnostics = pd.read_csv(
        RESULTS_DIR / "rebalance_diagnostics_10bps.csv",
        parse_dates=["date"],
    )
    dates = [args.date] if args.date else [
        date.date().isoformat() for date in diagnostics["date"].tail(2)
    ]
    graph = build_graph()
    notes: list[str] = []
    for date in dates:
        result = graph.invoke({"requested_date": date})
        notes.append(result["note_markdown"])
    combined = "# Notes de decision agentique\n\n" + "\n".join(notes)
    notes_path = RESULTS_DIR / "agent_decision_notes.md"
    notes_path.write_text(combined, encoding="utf-8")
    print(f"Workflow LangGraph execute pour {len(dates)} date(s).")
    print(f"Notes : {notes_path}")


if __name__ == "__main__":
    main()
