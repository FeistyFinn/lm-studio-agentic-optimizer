import json
from unittest.mock import MagicMock, patch

import pytest

from trial_runner import append_result, run_single_trial


@pytest.fixture
def telemetry():
    mock_tm = MagicMock()
    mock_tm.stop.return_value = 8.5
    mock_tm.baseline_vram_gb = 1.2
    with patch("trial_runner.TelemetryMonitor", return_value=mock_tm), patch(
        "trial_runner.time.sleep"
    ):
        yield mock_tm


def make_client(load_ok=True, metrics=None):
    client = MagicMock()
    client.transport = "sdk"
    client.load_model.return_value = load_ok
    client.generate_with_metrics.return_value = metrics or {
        "success": True,
        "ttft": 0.5,
        "tps": 30.0,
        "prompt_tokens": 12,
        "prefill_tps": 24.0,
        "output": "Hello world output",
    }
    return client


def test_success_result_shape(telemetry):
    result = run_single_trial("m", 2048, 0.5, "prompt", client=make_client())

    assert result["status"] == "Success"
    assert result["tps"] == 30.0
    assert result["prefill_tps"] == 24.0
    assert result["baseline_vram_gb"] == 1.2
    assert result["peak_vram_gb"] == 8.5
    assert result["transport"] == "sdk"
    assert "quality_score" not in result
    assert "vram_estimate_gb" not in result


def test_vram_estimate_fallback_when_no_telemetry(telemetry):
    telemetry.stop.return_value = None

    with patch("trial_runner.latest_load_estimate_gb", return_value=6.77):
        result = run_single_trial("m", 2048, 1.0, "prompt", client=make_client())

    assert result["peak_vram_gb"] is None
    assert result["vram_estimate_gb"] == 6.77


def test_oom_result(telemetry):
    result = run_single_trial("m", 8192, 1.0, "prompt", client=make_client(False))

    assert result["status"] == "OOM/Load Fail"
    assert result["tps"] == 0.0
    assert result["peak_vram_gb"] == 8.5


def test_generation_failure(telemetry):
    client = make_client(metrics={"success": False, "error": "timeout"})

    result = run_single_trial("m", 2048, 0.5, "prompt", client=client)

    assert result["status"] == "Generation Fail"
    assert result["error"] == "timeout"


def test_judge_scoring(telemetry):
    judge = MagicMock()
    judge.evaluate.return_value = {"score": 8, "reasoning": "x" * 500}

    result = run_single_trial(
        "m", 2048, 0.5, "prompt", judge=judge, rubric="rubric", client=make_client()
    )

    judge.evaluate.assert_called_once()
    assert result["quality_score"] == 8
    assert len(result["judge_reasoning"]) == 300


def test_judge_skipped_without_rubric(telemetry):
    judge = MagicMock()

    result = run_single_trial(
        "m", 2048, 0.5, "prompt", judge=judge, rubric=None, client=make_client()
    )

    judge.evaluate.assert_not_called()
    assert "quality_score" not in result


def test_append_result_creates_and_appends(tmp_path):
    path = tmp_path / "results.json"

    append_result(str(path), {"a": 1})
    append_result(str(path), {"b": 2})

    assert json.loads(path.read_text()) == [{"a": 1}, {"b": 2}]


def test_append_result_recovers_from_corrupt_file(tmp_path):
    path = tmp_path / "results.json"
    path.write_text("not json {")

    append_result(str(path), {"a": 1})

    assert json.loads(path.read_text()) == [{"a": 1}]
