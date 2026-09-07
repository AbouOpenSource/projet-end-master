from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import agent_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_workflow_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid_quality: bool = False,
) -> Path:
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    results_dir.mkdir()

    shutil.copy(PROJECT_ROOT / "data/quality_report.json", data_dir / "quality_report.json")
    for filename in (
        "rebalance_diagnostics_10bps.csv",
        "monthly_asset_returns.csv",
    ):
        shutil.copy(PROJECT_ROOT / "results" / filename, results_dir / filename)

    if invalid_quality:
        quality_path = data_dir / "quality_report.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        quality["missing_values_by_ticker"]["VLUE"] = 1
        quality_path.write_text(
            json.dumps(quality, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    monkeypatch.setattr(agent_workflow, "DATA_DIR", data_dir)
    monkeypatch.setattr(agent_workflow, "RESULTS_DIR", results_dir)
    return results_dir


def test_quality_gate_accepts_archived_quality_report() -> None:
    quality = json.loads(
        (PROJECT_ROOT / "data/quality_report.json").read_text(encoding="utf-8")
    )

    result = agent_workflow.quality_gate({"quality": quality})

    assert result == {"quality_ok": True, "quality_flags": []}


@pytest.mark.parametrize(
    ("field", "value", "expected_flag"),
    [
        ("duplicate_dates", 1, "dates dupliquees"),
        ("missing_values_by_ticker", {"VLUE": 1}, "valeurs manquantes"),
        ("non_positive_values_by_ticker", {"VLUE": 1}, "prix non positifs"),
    ],
)
def test_quality_gate_rejects_invalid_quality(
    field: str,
    value: object,
    expected_flag: str,
) -> None:
    quality = {
        "duplicate_dates": 0,
        "missing_values_by_ticker": {},
        "non_positive_values_by_ticker": {},
        field: value,
    }

    result = agent_workflow.quality_gate({"quality": quality})

    assert result["quality_ok"] is False
    assert result["quality_flags"] == [expected_flag]


def test_langgraph_valid_path_persists_supervised_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = prepare_workflow_inputs(tmp_path, monkeypatch)

    result = agent_workflow.build_graph().invoke({"requested_date": "2025-12-31"})

    assert result["quality_ok"] is True
    assert result["risk_status"] == "a valider"
    assert result["trace"]["permissions"] == {
        "can_change_weights": False,
        "can_send_orders": False,
        "human_validation_required": True,
    }
    assert (results_dir / "agent_decision_note_2025-12-31.md").exists()
    assert (results_dir / "agent_decision_trace_2025-12-31.json").exists()


def test_langgraph_invalid_quality_path_is_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_dir = prepare_workflow_inputs(
        tmp_path,
        monkeypatch,
        invalid_quality=True,
    )

    result = agent_workflow.build_graph().invoke({"requested_date": "2025-12-31"})

    assert result["quality_ok"] is False
    assert result["quality_flags"] == ["valeurs manquantes"]
    assert "bloque" in result["note_markdown"]
    assert "rebalance" not in result["trace"]
    assert (results_dir / "agent_decision_note_2025-12-31.md").exists()
    assert (results_dir / "agent_decision_trace_2025-12-31.json").exists()


def test_rebalance_analyst_rejects_unknown_date() -> None:
    diagnostics = pd.DataFrame(
        {"net_return": [0.01], "turnover": [0.02], "transaction_cost": [0.00002]},
        index=pd.DatetimeIndex(["2025-12-31"]),
    )
    returns = pd.DataFrame(
        {ticker: [0.0] for ticker in agent_workflow.FACTOR_TICKERS.values()},
        index=diagnostics.index,
    )

    with pytest.raises(ValueError, match="indisponible"):
        agent_workflow.rebalance_analyst(
            {
                "requested_date": "2025-11-30",
                "diagnostics": diagnostics,
                "monthly_returns": returns,
            }
        )
