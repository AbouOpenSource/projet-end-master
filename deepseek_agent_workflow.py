"""Workflow LangGraph comparant une synthese DeepSeek a une reference deterministe.

Le modele ne calcule pas les rendements, ne modifie pas les poids et ne peut pas
envoyer d'ordre. Il lit uniquement les sorties structurees du prototype puis sa
reponse est validee par des controles deterministes avant archivage.

Variables d'environnement :
    DEEPSEEK_API_KEY : cle API, jamais ecrite dans les resultats.
    DEEPSEEK_BASE_URL : URL compatible OpenAI, par defaut https://api.deepseek.com.
    DEEPSEEK_MODEL : modele DeepSeek, par defaut deepseek-v4-flash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agent_workflow import (
    FACTOR_TICKERS,
    WorkflowState,
    load_inputs,
    quality_gate,
    rebalance_analyst,
    risk_gate,
    route_quality,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PROMPT_VERSION = "deepseek-synthesis-v2"
load_dotenv(ROOT / ".env")


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def reproducibility_metadata() -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "data_hashes": {
            "data/quality_report.json": _sha256_file(ROOT / "data/quality_report.json"),
            "results/monthly_asset_returns.csv": _sha256_file(
                ROOT / "results/monthly_asset_returns.csv"
            ),
            "results/rebalance_diagnostics_10bps.csv": _sha256_file(
                ROOT / "results/rebalance_diagnostics_10bps.csv"
            ),
        },
    }


class DeepSeekState(WorkflowState, total=False):
    prompt_payload: dict[str, Any]
    llm_output: dict[str, Any]
    llm_raw_content: str
    llm_metadata: dict[str, Any]
    evaluation: dict[str, Any]
    validation_status: str
    rejection_reasons: list[str]
    note_output: dict[str, Any]
    reproducibility: dict[str, Any]
    error: str


def _client() -> tuple[Any, str, str]:
    """Create the OpenAI-compatible client without exposing the API key."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY est absente. Definissez-la dans l'environnement "
            "avant le test."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Le paquet openai est requis. Installez requirements.txt."
        ) from exc

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    return OpenAI(api_key=api_key, base_url=base_url), base_url, model


def build_prompt_payload(state: DeepSeekState) -> dict[str, Any]:
    """Build the only quantitative context exposed to the language model."""
    rebalance = state["rebalance"]
    return {
        "task": "Synthetiser un reequilibrage multifactoriel a partir de donnees deja calculees.",
        "date": state["date"],
        "quality_ok": state["quality_ok"],
        "quality_flags": state["quality_flags"],
        "rebalance": rebalance,
        "risk_status": state["risk_status"],
        "risk_alerts": state["risk_alerts"],
        "permissions": {
            "can_change_weights": False,
            "can_send_orders": False,
            "human_validation_required": True,
        },
        "constraints": [
            "Utiliser exclusivement les valeurs JSON fournies.",
            "Ne recalculer aucun rendement et ne pas inventer de chiffre.",
            "Ne modifier aucune ponderation et ne generer aucun ordre.",
            "Signaler les limites et exiger la validation humaine.",
        ],
    }


def llm_synthesis(state: DeepSeekState) -> dict[str, Any]:
    """Call DeepSeek with a constrained JSON contract."""
    client, base_url, model = _client()
    payload = build_prompt_payload(state)
    system_prompt = (
        "Tu es un assistant de documentation pour un desk de gestion supervise. "
        "Tu interpretes uniquement des sorties quantitatives deja calculees. "
        "Tu ne calcules pas, ne modifies pas les poids, ne proposes pas d'ordre et "
        "ne remplaces pas la validation humaine. Reponds exclusivement avec un "
        "objet JSON valide, sans Markdown, conforme a ce contrat : "
        "summary (string), reported_values (object reprenant factor_returns, "
        "net_return, turnover et transaction_cost exactement ou avec un arrondi "
        "visible), recommendation (string), limitations (array de strings), "
        "human_validation_required (boolean), can_change_weights (boolean), "
        "can_send_orders (boolean), cited_fields (array de strings)."
    )
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=3000,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    content = response.choices[0].message.content or ""
    parse_error = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        # The graph must persist and reject malformed model output instead of
        # turning a contract violation into an untraceable workflow crash.
        parsed = {}
        parse_error = f"JSON invalide : {exc}"
    usage = getattr(response, "usage", None)
    prompt_material = json.dumps(
        {"system_prompt": system_prompt, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    metadata = {
        "provider": "deepseek",
        "model": getattr(response, "model", model),
        "base_url": base_url,
        "request_id": getattr(response, "id", None),
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(prompt_material).hexdigest(),
        "system_prompt": system_prompt,
        "parse_error": parse_error,
    }
    return {
        "llm_output": parsed,
        "llm_raw_content": content,
        "llm_metadata": metadata,
        "prompt_payload": payload,
    }


def _number_close(actual: Any, expected: float, tolerance: float = 1e-4) -> bool:
    try:
        return abs(float(actual) - expected) <= tolerance
    except (TypeError, ValueError):
        return False


def validate_llm_output(state: DeepSeekState) -> dict[str, Any]:
    """Score the model against the deterministic source of truth."""
    raw_output = state.get("llm_output")
    output = raw_output if isinstance(raw_output, dict) else {}
    expected = state["rebalance"]
    reported = output.get("reported_values", {})
    reported = reported if isinstance(reported, dict) else {}
    expected_factors = expected["factor_returns"]
    reported_factors = reported.get("factor_returns", {})
    reported_factors = reported_factors if isinstance(reported_factors, dict) else {}
    factor_checks = {
        label: _number_close(reported_factors.get(label), value)
        for label, value in expected_factors.items()
    }
    numeric_values = [
        _number_close(reported.get("net_return"), expected["net_return"]),
        _number_close(reported.get("turnover"), expected["turnover"]),
        _number_close(reported.get("transaction_cost"), expected["transaction_cost"]),
        *factor_checks.values(),
    ]
    numeric_fidelity = sum(numeric_values) / len(numeric_values)
    limitations = output.get("limitations", [])
    limitations = limitations if isinstance(limitations, list) else []
    cited_fields = output.get("cited_fields", [])
    cited_fields = cited_fields if isinstance(cited_fields, list) else []
    required_fields = {
        "summary": isinstance(output.get("summary"), str) and bool(output["summary"].strip()),
        "recommendation": isinstance(output.get("recommendation"), str)
        and bool(output["recommendation"].strip()),
        "limitations": len(limitations) >= 2,
        "cited_fields": len(cited_fields) >= 3,
    }
    completeness = sum(required_fields.values()) / len(required_fields)
    permissions_ok = (
        output.get("human_validation_required") is True
        and output.get("can_change_weights") is False
        and output.get("can_send_orders") is False
    )
    limitations_text = " ".join(str(item).lower() for item in limitations)
    limitations_disclosed = any(
        marker in limitations_text for marker in ("limite", "liquid", "slippage", "validation")
    )
    rejection_reasons: list[str] = []
    if numeric_fidelity < 1:
        rejection_reasons.append("valeurs numériques non conformes")
    if completeness < 1:
        rejection_reasons.append("contrat JSON incomplet")
    if not permissions_ok:
        rejection_reasons.append("permissions incompatibles avec le contrat")
    if not limitations_disclosed:
        rejection_reasons.append("limites ou validation humaine non divulguées")
    accepted = not rejection_reasons
    evaluation = {
        "numeric_fidelity": round(numeric_fidelity, 4),
        "numeric_checks": factor_checks
        | {
            "net_return": numeric_values[0],
            "turnover": numeric_values[1],
            "transaction_cost": numeric_values[2],
        },
        "completeness": round(completeness, 4),
        "permission_compliance": float(permissions_ok),
        "limitations_disclosed": float(limitations_disclosed),
        "rejection_reasons": rejection_reasons,
        "accepted": accepted,
    }
    return {
        "evaluation": evaluation,
        "validation_status": "accepted" if accepted else "rejected",
    }


def rejected_llm(state: DeepSeekState) -> dict[str, Any]:
    """Create a rejected trace when the model output fails deterministic checks."""
    reasons = list(state.get("evaluation", {}).get("rejection_reasons", []))
    if not reasons:
        reasons = ["sortie LLM non conforme"]
    note_output = {
        "summary": "La sortie DeepSeek est rejetée par le contrôle déterministe.",
        "recommendation": "Aucune recommandation exploitable : une revue humaine est obligatoire.",
        "limitations": reasons,
    }
    return {
        "validation_status": "rejected",
        "rejection_reasons": reasons,
        "note_output": note_output,
    }


def blocked_llm(state: DeepSeekState) -> dict[str, Any]:
    """Create a blocked trace without calling the model when data quality fails."""
    date = state.get("requested_date") or "date inconnue"
    output = {
        "summary": "La synthèse DeepSeek est bloquée car le contrôle qualité a échoué.",
        "reported_values": {},
        "recommendation": "Aucune décision : corriger les données puis relancer.",
        "limitations": state["quality_flags"],
        "human_validation_required": True,
        "can_change_weights": False,
        "can_send_orders": False,
        "cited_fields": ["quality_flags"],
    }
    return {
        "date": date,
        "llm_output": output,
        "llm_metadata": {"provider": "deepseek", "skipped": True},
        "evaluation": {
            "numeric_fidelity": 0.0,
            "completeness": 1.0,
            "permission_compliance": 1.0,
            "limitations_disclosed": 1.0,
            "rejection_reasons": ["contrôle qualité bloqué"],
            "accepted": False,
        },
        "validation_status": "blocked",
        "note_output": output,
    }


def persist_llm(state: DeepSeekState) -> dict[str, Any]:
    date = state.get("date") or state.get("requested_date") or "date inconnue"
    note_output = state.get("note_output") or state.get("llm_output") or {}
    if not isinstance(note_output, dict):
        note_output = {}
    limitations = note_output.get("limitations", [])
    if not isinstance(limitations, list):
        limitations = [str(limitations)]
    trace = {
        "workflow": "langgraph_deepseek_supervised",
        "date": date,
        "quality_ok": state["quality_ok"],
        "quality_flags": state["quality_flags"],
        "risk_status": state.get("risk_status", "bloqué"),
        "risk_alerts": state.get("risk_alerts", []),
        "rebalance": state.get("rebalance"),
        "llm_output": state.get("llm_output"),
        "llm_raw_content": state.get("llm_raw_content"),
        "llm_metadata": state.get("llm_metadata", {}),
        "prompt_payload": state.get("prompt_payload"),
        "evaluation": state.get("evaluation", {}),
        "validation_status": state.get("validation_status", "blocked"),
        "rejection_reasons": state.get("rejection_reasons", []),
        "reproducibility": reproducibility_metadata(),
        "permissions": {
            "can_change_weights": False,
            "can_send_orders": False,
            "human_validation_required": True,
        },
    }
    summary = str(note_output.get("summary", "Aucune synthèse exploitable n'a été produite."))
    recommendation = str(note_output.get("recommendation", "Aucune recommandation exploitable."))
    note = (
        f"# Note DeepSeek du {date}\n\n"
        f"{summary}\n\n"
        f"**Recommandation :** {recommendation}\n\n"
        "**Limites signalées :**\n"
        + "\n".join(f"- {item}" for item in limitations)
        + "\n\n"
        f"**Statut du contrôle :** {trace['validation_status']}\n\n"
        f"**Évaluation automatique :** {json.dumps(state.get('evaluation', {}), ensure_ascii=False)}\n"
    )
    (RESULTS_DIR / f"deepseek_decision_trace_{date}.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / f"deepseek_decision_note_{date}.md").write_text(note, encoding="utf-8")
    return {"trace": trace}


def build_graph():
    builder = StateGraph(DeepSeekState)
    builder.add_node("load_inputs", load_inputs)
    builder.add_node("quality_gate", quality_gate)
    builder.add_node("rebalance_analyst", rebalance_analyst)
    builder.add_node("risk_gate", risk_gate)
    builder.add_node("llm_synthesis", llm_synthesis)
    builder.add_node("validate_llm_output", validate_llm_output)
    builder.add_node("rejected_llm", rejected_llm)
    builder.add_node("blocked_llm", blocked_llm)
    builder.add_node("persist_llm", persist_llm)
    builder.add_edge(START, "load_inputs")
    builder.add_edge("load_inputs", "quality_gate")
    builder.add_conditional_edges(
        "quality_gate",
        route_quality,
        {"rebalance": "rebalance_analyst", "blocked": "blocked_llm"},
    )
    builder.add_edge("rebalance_analyst", "risk_gate")
    builder.add_edge("risk_gate", "llm_synthesis")
    builder.add_edge("llm_synthesis", "validate_llm_output")
    builder.add_conditional_edges(
        "validate_llm_output",
        lambda state: "persist" if state["validation_status"] == "accepted" else "rejected",
        {"persist": "persist_llm", "rejected": "rejected_llm"},
    )
    builder.add_edge("rejected_llm", "persist_llm")
    builder.add_edge("blocked_llm", "persist_llm")
    builder.add_edge("persist_llm", END)
    return builder.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="date mensuelle YYYY-MM-DD")
    parser.add_argument(
        "--all-last-two",
        action="store_true",
        help="execute les deux dates de demonstration du rapport",
    )
    args = parser.parse_args()
    diagnostics = pd.read_csv(RESULTS_DIR / "rebalance_diagnostics_10bps.csv", parse_dates=["date"])
    if args.date:
        dates = [args.date]
    elif args.all_last_two:
        dates = [date.date().isoformat() for date in diagnostics["date"].tail(2)]
    else:
        dates = [diagnostics["date"].iloc[-1].date().isoformat()]
    graph = build_graph()
    evaluations = []
    for date in dates:
        result = graph.invoke({"requested_date": date})
        evaluations.append(
            {
                "date": date,
                "workflow": "deepseek",
                **result["evaluation"],
                "model": result["llm_metadata"].get("model"),
                "elapsed_ms": result["llm_metadata"].get("elapsed_ms"),
            }
        )
    comparison_path = RESULTS_DIR / "deepseek_comparison.csv"
    pd.DataFrame(evaluations).to_csv(comparison_path, index=False)
    print(f"Workflow DeepSeek execute pour {len(dates)} date(s).")
    print(f"Evaluation : {comparison_path}")


if __name__ == "__main__":
    main()
