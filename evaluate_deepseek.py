"""Evaluation rapide et reproductible de la couche DeepSeek.

Le moteur quantitatif et les controles deterministes restent la reference.
Le script mesure la fidelite numerique, les permissions, la divulgation des
risques, le routage des cas bloques et la stabilite des sorties. Il ne mesure
pas la capacite du modele a predire les rendements.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import agent_workflow
import deepseek_agent_workflow

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "deepseek_evaluation.json"

EXPECTED_REPORTED = {"factor_returns", "net_return", "turnover", "transaction_cost"}
ALLOWED_CITED = EXPECTED_REPORTED | {"date", "quality_ok", "quality_flags", "risk_alerts", "risk_status", "target_weights", "permissions", "can_change_weights", "can_send_orders", "human_validation_required", "constraints", "task"}
REQUIRED_FIELDS = {
    "summary", "reported_values", "recommendation", "limitations",
    "human_validation_required", "can_change_weights", "can_send_orders",
    "cited_fields",
}


@dataclass
class EvaluationCase:
    case_id: str
    kind: str
    state: dict[str, Any]
    expects_llm: bool


def _dates(index: pd.Index, count: int) -> list[str]:
    values = [pd.Timestamp(value).date().isoformat() for value in index]
    if not values or count <= 0:
        raise ValueError("Le nombre de cas et les dates doivent etre valides.")
    if count >= len(values):
        return values
    if count == 1:
        return [values[-1]]
    positions = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
    return [values[position] for position in positions]


def _state(inputs: dict[str, Any], date: str) -> dict[str, Any]:
    state = {"requested_date": date, **copy.deepcopy(inputs)}
    state.update(agent_workflow.quality_gate(state))
    if not state["quality_ok"]:
        raise ValueError("Les donnees de reference sont invalides.")
    state.update(agent_workflow.rebalance_analyst(state))
    state.update(agent_workflow.risk_gate(state))
    return state


def _risk_stress(state: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(state)
    rebalance = result["rebalance"]
    gross = float(rebalance["net_return"]) + float(rebalance["transaction_cost"])
    rebalance["turnover"] = 0.15
    rebalance["transaction_cost"] = 0.001 * rebalance["turnover"]
    rebalance["net_return"] = gross - rebalance["transaction_cost"]
    result["risk_alerts"] = [
        *result.get("risk_alerts", []),
        "turnover superieur a 10 % : escalade requise",
    ]
    return result


def build_evaluation_cases(valid_cases: int = 10, include_blocked: bool = True) -> list[EvaluationCase]:
    inputs = agent_workflow.load_inputs({})
    cases = [
        EvaluationCase(
            case_id=f"valid_{i:02d}",
            kind="valid",
            state=_state(inputs, date),
            expects_llm=True,
        )
        for i, date in enumerate(_dates(inputs["diagnostics"].index, valid_cases), start=1)
    ]
    cases.append(
        EvaluationCase(
            case_id="risk_high_turnover",
            kind="risk_stress",
            state=_risk_stress(cases[-1].state),
            expects_llm=True,
        )
    )
    if include_blocked:
        blocked = copy.deepcopy(cases[0].state)
        blocked["quality_ok"] = False
        blocked["quality_flags"] = ["valeurs manquantes"]
        blocked.pop("rebalance", None)
        blocked.pop("risk_status", None)
        blocked.pop("risk_alerts", None)
        cases.append(EvaluationCase("quality_blocked", "quality_blocked", blocked, False))
    return cases


def _score_consistency(state: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    reported = output.get("reported_values", {})
    reported = reported if isinstance(reported, dict) else {}
    cited = output.get("cited_fields", [])
    cited = cited if isinstance(cited, list) else []
    unexpected = sorted(set(reported) - EXPECTED_REPORTED)
    invalid_citations = sorted({str(value) for value in cited if str(value) not in ALLOWED_CITED and not any(str(value).startswith(prefix) for prefix in ("rebalance.", "permissions.", "factor_returns."))})
    text = " ".join([
        str(output.get("summary", "")),
        str(output.get("recommendation", "")),
        " ".join(str(item) for item in output.get("limitations", [])),
    ]).lower()
    alerts = state.get("risk_alerts", [])
    risk_disclosed = not alerts or "risk_alerts" in cited or any(
        marker in text for marker in ("validation", "risque", "liquid", "slippage", "escalade")
    )
    return {
        "unexpected_reported_fields": unexpected,
        "invalid_cited_fields": invalid_citations,
        "structured_consistency": float(not unexpected and not invalid_citations),
        "risk_disclosed": float(risk_disclosed),
    }


def score_model_output(state: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    evaluation = dict(
        deepseek_agent_workflow.validate_llm_output(
            {**state, "llm_output": output}
        )["evaluation"]
    )
    missing = sorted(REQUIRED_FIELDS - set(output))
    evaluation.update(_score_consistency(state, output))
    evaluation["missing_fields"] = missing
    evaluation["json_contract"] = float(not missing)
    evaluation["accepted"] = bool(
        evaluation["accepted"]
        and not missing
        and evaluation["structured_consistency"] == 1.0
        and evaluation["risk_disclosed"] == 1.0
    )
    evaluation["critical_failure"] = bool(
        evaluation["numeric_fidelity"] < 1.0
        or evaluation["permission_compliance"] < 1.0
        or evaluation["json_contract"] < 1.0
        or evaluation["structured_consistency"] < 1.0
    )
    return evaluation


def _normalize(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _projection(output: dict[str, Any]) -> str:
    selected = {
        "reported_values": _normalize(output.get("reported_values")),
        "human_validation_required": output.get("human_validation_required"),
        "can_change_weights": output.get("can_change_weights"),
        "can_send_orders": output.get("can_send_orders"),
    }
    return json.dumps(selected, ensure_ascii=False, sort_keys=True)


def _citation_projection(output: dict[str, Any]) -> str:
    return json.dumps(
        sorted(set(output.get("cited_fields", []))),
        ensure_ascii=False,
    )


def _call_with_retry(state: dict[str, Any], attempts: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return deepseek_agent_workflow.llm_synthesis(copy.deepcopy(state))
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def evaluate_case(
    case: EvaluationCase,
    repeats: int = 1,
    synthesis: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats doit etre positif.")
    if not case.expects_llm:
        result = deepseek_agent_workflow.blocked_llm(case.state)
        return {
            "case_id": case.case_id,
            "kind": case.kind,
            "model_called": False,
            "case_passed": True,
            "gate_compliance": 1.0,
            "evaluation": result["evaluation"],
            "llm_output": result["llm_output"],
        }

    call = synthesis or _call_with_retry
    repetitions, projections, citation_projections = [], [], []
    for repeat in range(1, repeats + 1):
        state = copy.deepcopy(case.state)
        try:
            result = call(state)
            output = result["llm_output"]
            evaluation = score_model_output(state, output)
            repetitions.append({
                "repeat": repeat,
                "evaluation": evaluation,
                "llm_output": output,
                "metadata": result.get("llm_metadata", {}),
            })
            projections.append(_projection(output))
            citation_projections.append(_citation_projection(output))
        except Exception as exc:
            repetitions.append({
                "repeat": repeat,
                "error": f"{type(exc).__name__}: {exc}",
                "evaluation": {"accepted": False, "critical_failure": True},
            })

    successful = [item for item in repetitions if "error" not in item]
    values = [float(item["evaluation"].get("numeric_fidelity", 0.0)) for item in successful]
    accepted = [bool(item["evaluation"].get("accepted", False)) for item in successful]
    first = successful[0]["evaluation"] if successful else {"accepted": False}
    passed = bool(
        len(successful) == repeats
        and all(accepted)
        and all(not item["evaluation"].get("critical_failure", True) for item in successful)
    )
    return {
        "case_id": case.case_id,
        "kind": case.kind,
        "model_called": True,
        "repeats": repeats,
        "case_passed": passed,
        "stability": float(len(set(projections)) <= 1) if projections else 0.0,
        "citation_stability": float(len(set(citation_projections)) <= 1) if citation_projections else 0.0,
        "numeric_fidelity": round(sum(values) / len(values), 4) if values else 0.0,
        "accepted_rate": round(sum(accepted) / len(accepted), 4) if accepted else 0.0,
        "evaluation": first,
        "repetitions": repetitions,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    model = [item for item in records if item["model_called"]]
    blocked = [item for item in records if not item["model_called"]]
    repetitions = [
        repetition
        for item in model
        for repetition in item.get("repetitions", [])
        if "error" not in repetition
    ]
    evaluations = [item.get("evaluation", {}) for item in repetitions]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "cases_total": len(records),
        "model_cases": len(model),
        "blocked_cases": len(blocked),
        "case_pass_rate": mean([float(item["case_passed"]) for item in records]),
        "numeric_fidelity_mean": mean([
            float(item.get("evaluation", {}).get("numeric_fidelity", 0.0))
            for item in repetitions
        ]),
        "numeric_exact_rate": mean([
            float(item.get("evaluation", {}).get("numeric_fidelity", 0.0) == 1.0)
            for item in repetitions
        ]),
        "accepted_rate": mean([
            float(item.get("evaluation", {}).get("accepted", False))
            for item in repetitions
        ]),
        "permission_compliance_rate": mean([
            float(item.get("permission_compliance", 0.0))
            for item in evaluations
        ]),
        "structured_consistency_rate": mean([
            float(item.get("structured_consistency", 0.0))
            for item in evaluations
        ]),
        "risk_disclosure_rate": mean([
            float(item.get("risk_disclosed", 0.0))
            for item in evaluations
        ]),
        "stability_rate": mean([float(item.get("stability", 1.0)) for item in model]),
        "citation_stability_rate": mean([
            float(item.get("citation_stability", 1.0)) for item in model
        ]),
        "blocked_route_rate": mean([
            float(item.get("gate_compliance", 0.0)) for item in blocked
        ]),
        "critical_failure_count": sum(
            bool(item.get("critical_failure", False)) for item in evaluations
        ),
        "error_count": sum(
            "error" in repetition
            for item in model
            for repetition in item.get("repetitions", [])
        ),
    }


def write_report(output: Path, configuration: dict[str, Any], records: list[dict[str, Any]]) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "summary": summarize(records),
        "cases": records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for item in records:
        evaluation = item.get("evaluation", {})
        rows.append({
            "case_id": item["case_id"],
            "kind": item["kind"],
            "model_called": item["model_called"],
            "case_passed": item["case_passed"],
            "numeric_fidelity": item.get("numeric_fidelity", evaluation.get("numeric_fidelity")),
            "accepted_rate": item.get("accepted_rate"),
            "permission_compliance": evaluation.get("permission_compliance"),
            "structured_consistency": evaluation.get("structured_consistency"),
            "risk_disclosed": evaluation.get("risk_disclosed"),
            "stability": item.get("stability", 1.0),
            "citation_stability": item.get("citation_stability", 1.0),
            "critical_failure": evaluation.get("critical_failure", False),
        })
    csv = output.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    return output, csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-cases", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-blocked", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = build_evaluation_cases(args.valid_cases, not args.no_blocked)
    configuration = {
        "valid_cases": args.valid_cases,
        "repeats": args.repeats,
        "blocked_case": not args.no_blocked,
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    }
    if args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration,
            "cases": [
                {"case_id": item.case_id, "kind": item.kind, "expects_llm": item.expects_llm}
                for item in cases
            ],
            "note": "Aucun appel DeepSeek n'a ete effectue.",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Benchmark prepare sans appel API : {args.output}")
        return

    records = [evaluate_case(item, args.repeats) for item in cases]
    json_path, csv_path = write_report(args.output, configuration, records)
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
    print(f"Rapport JSON : {json_path}")
    print(f"Rapport CSV  : {csv_path}")


if __name__ == "__main__":
    main()
