from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import agent_workflow
import deepseek_agent_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_deepseek_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid_quality: bool = False,
) -> Path:
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    results_dir.mkdir()

    quality = json.loads(
        (PROJECT_ROOT / "data/quality_report.json").read_text(encoding="utf-8")
    )
    if invalid_quality:
        quality["missing_values_by_ticker"]["VLUE"] = 1
    (data_dir / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for filename in (
        "rebalance_diagnostics_10bps.csv",
        "monthly_asset_returns.csv",
    ):
        shutil.copy(PROJECT_ROOT / "results" / filename, results_dir / filename)

    monkeypatch.setattr(agent_workflow, "DATA_DIR", data_dir)
    monkeypatch.setattr(agent_workflow, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(deepseek_agent_workflow, "RESULTS_DIR", results_dir)
    return results_dir


def valid_llm_state() -> deepseek_agent_workflow.DeepSeekState:
    rebalance = {
        "date": "2025-12-31",
        "factor_returns": {
            "Value": 0.01,
            "Momentum": -0.02,
            "Quality": 0.03,
            "Low volatility": 0.04,
        },
        "net_return": 0.015,
        "turnover": 0.02,
        "transaction_cost": 0.00002,
    }
    return {
        "rebalance": rebalance,
        "llm_output": {
            "summary": "Résumé fidèle.",
            "reported_values": {
                "factor_returns": rebalance["factor_returns"],
                "net_return": rebalance["net_return"],
                "turnover": rebalance["turnover"],
                "transaction_cost": rebalance["transaction_cost"],
            },
            "recommendation": "Conserver sous validation humaine.",
            "limitations": [
                "Validation humaine obligatoire.",
                "Liquidité et slippage non modélisés.",
            ],
            "human_validation_required": True,
            "can_change_weights": False,
            "can_send_orders": False,
            "cited_fields": ["factor_returns", "net_return", "risk_alerts"],
        },
    }


def test_validate_llm_output_accepts_faithful_supervised_response() -> None:
    state = valid_llm_state()
    result = deepseek_agent_workflow.validate_llm_output(state)

    assert result["evaluation"]["numeric_fidelity"] == 1.0
    assert result["evaluation"]["completeness"] == 1.0
    assert result["evaluation"]["permission_compliance"] == 1.0
    assert result["evaluation"]["limitations_disclosed"] == 1.0
    assert result["evaluation"]["accepted"] is True


def test_deepseek_blocked_path_persists_without_calling_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = prepare_deepseek_inputs(
        tmp_path,
        monkeypatch,
        invalid_quality=True,
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    result = deepseek_agent_workflow.build_graph().invoke(
        {"requested_date": "2025-12-31"}
    )

    assert result["quality_ok"] is False
    assert result["evaluation"]["accepted"] is False
    assert result["llm_metadata"]["skipped"] is True
    assert result["trace"]["risk_status"] == "bloque"
    assert result["trace"]["rebalance"] is None
    assert (results_dir / "deepseek_decision_note_2025-12-31.md").exists()
    assert (results_dir / "deepseek_decision_trace_2025-12-31.json").exists()
