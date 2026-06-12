from unittest.mock import MagicMock, patch

import pytest

from optimizer_agent import HardwareOptimizerAgent


@pytest.fixture
def mock_client_class():
    with patch("optimizer_agent.LMStudioHardwareClient") as mock:
        yield mock


def make_trial(context_length, gpu_ratio):
    """objective() calls suggest_categorical twice: context_length, gpu_ratio."""
    mock_trial = MagicMock()
    mock_trial.suggest_categorical.side_effect = [context_length, gpu_ratio]
    return mock_trial


def test_objective_load_failure(mock_client_class):
    mock_instance = MagicMock()
    mock_instance.load_model.return_value = False
    mock_client_class.return_value = mock_instance

    agent = HardwareOptimizerAgent(dataset_path="dummy.json")
    agent.dataset = [{"prompt": "test"}]

    reward = agent.objective(make_trial(8192, 1.0))

    assert reward == 0.0
    assert agent.current_trial_stats["status"] == "OOM/Load Fail"


def test_objective_success(mock_client_class):
    mock_instance = MagicMock()
    mock_instance.load_model.return_value = True
    mock_instance.generate_with_metrics.return_value = {
        "success": True,
        "tps": 25.0,
        "ttft": 1.5,
        "output": "Test summary",
    }
    mock_client_class.return_value = mock_instance

    agent = HardwareOptimizerAgent(dataset_path="dummy.json")
    agent.dataset = [{"prompt": "test"}]

    reward = agent.objective(make_trial(4096, 0.5))

    assert reward > 0.0
    assert agent.current_trial_stats["status"] == "Success"
    assert agent.current_trial_stats["tps"] == 25.0


def test_tps_reward_ceiling_env(mock_client_class, monkeypatch):
    monkeypatch.setenv("TPS_REWARD_CEILING", "100.0")
    mock_client_class.return_value = MagicMock()

    agent = HardwareOptimizerAgent(dataset_path="dummy.json")

    assert agent.tps_reward_ceiling == 100.0


def test_run_optimization(mock_client_class, tmp_path, monkeypatch):
    # Just ensure the loop runs without crashing
    mock_instance = MagicMock()
    mock_instance.load_model.return_value = True
    mock_instance.generate_with_metrics.return_value = {
        "success": True,
        "tps": 10.0,
        "ttft": 2.0,
        "output": "Test",
    }
    mock_client_class.return_value = mock_instance

    agent = HardwareOptimizerAgent(dataset_path="dummy.json")
    agent.dataset = [{"prompt": "test"}]

    # run_optimization writes optimal_settings.json to the cwd
    monkeypatch.chdir(tmp_path)
    best_trial = agent.run_optimization(n_trials=2)

    assert "context_length" in best_trial.params
    assert "gpu_ratio" in best_trial.params
    assert (tmp_path / "optimal_settings.json").exists()
