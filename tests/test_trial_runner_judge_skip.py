from unittest.mock import MagicMock, patch

import pytest

from trial_runner import run_single_trial


@pytest.fixture
def telemetry():
    mock_tm = MagicMock()
    mock_tm.stop.return_value = 8.5
    mock_tm.baseline_vram_gb = 1.2
    with patch("trial_runner.TelemetryMonitor", return_value=mock_tm), patch(
        "trial_runner.time.sleep"
    ):
        yield mock_tm


def make_client(metrics):
    client = MagicMock()
    client.transport = "sdk"
    client.load_model.return_value = True
    client.generate_with_metrics.return_value = metrics
    return client


def test_judge_skipped_when_reasoning_eats_budget(telemetry):
    judge = MagicMock()
    client = make_client(
        {
            "success": True,
            "ttft": 5.0,
            "tps": 20.0,
            "output": "",
            "reasoning_chars": 2048,
        }
    )

    result = run_single_trial(
        "m", 2048, 1.0, "prompt", judge=judge, rubric="rubric", client=client
    )

    judge.evaluate.assert_not_called()
    assert result["quality_score"] is None
    assert "reasoning consumed the token budget" in result["judge_reasoning"]
    assert result["reasoning_chars"] == 2048


def test_judge_runs_on_truly_empty_output(telemetry):
    judge = MagicMock()
    judge.evaluate.return_value = {"score": 1, "reasoning": "empty"}
    client = make_client(
        {"success": True, "ttft": 1.0, "tps": 5.0, "output": "", "reasoning_chars": 0}
    )

    result = run_single_trial(
        "m", 2048, 1.0, "prompt", judge=judge, rubric="rubric", client=client
    )

    judge.evaluate.assert_called_once()
    assert result["quality_score"] == 1
