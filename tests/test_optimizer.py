import pytest
from unittest.mock import patch, MagicMock
from optimizer_agent import HardwareOptimizerAgent

@pytest.fixture
def mock_client_class():
    with patch("optimizer_agent.LMStudioHardwareClient") as mock:
        yield mock

def test_objective_load_failure(mock_client_class):
    # Setup mock client
    mock_instance = MagicMock()
    mock_instance.load_model.return_value = False
    mock_client_class.return_value = mock_instance

    agent = HardwareOptimizerAgent(dataset_path="dummy.json")
    agent.dataset = [{"prompt": "test"}]

    mock_trial = MagicMock()
    mock_trial.suggest_categorical.return_value = 8192
    mock_trial.suggest_float.return_value = 1.0

    reward = agent.objective(mock_trial)

    assert reward == 0.0
    assert agent.current_trial_stats["status"] == "OOM/Load Fail"

def test_objective_success(mock_client_class):
    # Setup mock client
    mock_instance = MagicMock()
    mock_instance.load_model.return_value = True
    mock_instance.generate_with_metrics.return_value = {
        "success": True,
        "tps": 25.0,
        "ttft": 1.5,
        "output": "Test summary"
    }
    mock_client_class.return_value = mock_instance

    agent = HardwareOptimizerAgent(dataset_path="dummy.json")
    agent.dataset = [{"prompt": "test"}]

    mock_trial = MagicMock()
    mock_trial.suggest_categorical.return_value = 4096
    mock_trial.suggest_float.return_value = 0.5

    reward = agent.objective(mock_trial)

    assert reward > 0.0
    assert agent.current_trial_stats["status"] == "Success"
    assert agent.current_trial_stats["tps"] == 25.0

def test_run_optimization(mock_client_class):
    # Just ensure the loop runs without crashing
    mock_instance = MagicMock()
    mock_instance.load_model.return_value = True
    mock_instance.generate_with_metrics.return_value = {
        "success": True, "tps": 10.0, "ttft": 2.0, "output": "Test"
    }
    mock_client_class.return_value = mock_instance

    agent = HardwareOptimizerAgent(dataset_path="dummy.json")
    agent.dataset = [{"prompt": "test"}]

    # We can mock the optuna callback or just let it run
    # Optuna handles MagicMock trials internally if we mock the whole optuna,
    # but we can also just run it with a very small n_trials
    best_params = agent.run_optimization(n_trials=2)

    assert "context_length" in best_params
    assert "gpu_ratio" in best_params
