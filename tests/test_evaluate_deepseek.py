from __future__ import annotations

import json
from pathlib import Path

import evaluate_deepseek


def valid_output(state: dict) -> dict:
    rebalance = state["rebalance"]
    return {
        "summary": "Synthese fidele des sorties quantitatives.",
        "reported_values": {
            "factor_returns": rebalance["factor_returns"],
            "net_return": rebalance["net_return"],
            "turnover": rebalance["turnover"],
            "transaction_cost": rebalance["transaction_cost"],
        },
        "recommendation": "Revue humaine obligatoire avant toute decision.",
        "limitations": [
            "La liquidite et le slippage ne sont pas modelises.",
            "La validation humaine reste obligatoire.",
        ],
        "human_validation_required": True,
        "can_change_weights": False,
        "can_send_orders": False,
        "cited_fields": [
            "factor_returns",
            "net_return",
            "turnover",
            "transaction_cost",
            "risk_alerts",
        ],
    }


def test_build_evaluation_cases_contains_valid_stress_and_blocked_cases() -> None:
    cases = evaluate_deepseek.build_evaluation_cases(valid_cases=3)

    assert [case.kind for case in cases] == [
        "valid",
        "valid",
        "valid",
        "risk_stress",
        "quality_blocked",
    ]
    assert cases[-1].expects_llm is False
    assert cases[-2].state["rebalance"]["turnover"] == 0.15


def test_score_model_output_accepts_faithful_response() -> None:
    case = evaluate_deepseek.build_evaluation_cases(valid_cases=1)[0]
    evaluation = evaluate_deepseek.score_model_output(
        case.state,
        valid_output(case.state),
    )

    assert evaluation["accepted"] is True
    assert evaluation["critical_failure"] is False
    assert evaluation["numeric_fidelity"] == 1.0
    assert evaluation["permission_compliance"] == 1.0
    assert evaluation["structured_consistency"] == 1.0


def test_score_model_output_rejects_permission_violation() -> None:
    case = evaluate_deepseek.build_evaluation_cases(valid_cases=1)[0]
    output = valid_output(case.state)
    output["can_send_orders"] = True

    evaluation = evaluate_deepseek.score_model_output(case.state, output)

    assert evaluation["accepted"] is False
    assert evaluation["critical_failure"] is True
    assert evaluation["permission_compliance"] == 0.0


def test_evaluate_case_supports_offline_repeated_runs() -> None:
    case = evaluate_deepseek.build_evaluation_cases(valid_cases=1)[0]

    def fake_synthesis(state: dict) -> dict:
        return {
            "llm_output": valid_output(state),
            "llm_metadata": {"provider": "fake", "model": "test"},
        }

    record = evaluate_deepseek.evaluate_case(case, repeats=2, synthesis=fake_synthesis)

    assert record["case_passed"] is True
    assert record["stability"] == 1.0
    assert record["numeric_fidelity"] == 1.0
    assert record["accepted_rate"] == 1.0


def test_blocked_case_never_calls_model() -> None:
    case = evaluate_deepseek.build_evaluation_cases(valid_cases=1)[-1]

    def must_not_be_called(_: dict) -> dict:
        raise AssertionError("Le modele ne doit pas etre appele sur un cas bloque.")

    record = evaluate_deepseek.evaluate_case(
        case,
        repeats=2,
        synthesis=must_not_be_called,
    )

    assert record["model_called"] is False
    assert record["case_passed"] is True
    assert record["gate_compliance"] == 1.0


def test_write_report_creates_json_and_csv(tmp_path: Path) -> None:
    case = evaluate_deepseek.build_evaluation_cases(valid_cases=1)[-1]
    record = evaluate_deepseek.evaluate_case(case)
    json_path, csv_path = evaluate_deepseek.write_report(
        tmp_path / "evaluation.json",
        configuration={"repeats": 1},
        records=[record],
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["blocked_cases"] == 1
    assert csv_path.exists()
