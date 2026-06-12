import optuna
import json
import os
import time
from typing import Any, Dict, List, Optional, Callable
from lm_studio_client import LMStudioHardwareClient

class HardwareOptimizerAgent:
    def __init__(self, dataset_path: str = "dataset.json") -> None:
        self.client: LMStudioHardwareClient = LMStudioHardwareClient()
        self.model: str = os.getenv("TARGET_MODEL", "local-model")

        self.dataset: List[Dict[str, str]]
        try:
            with open(dataset_path, "r") as f:
                self.dataset = json.load(f)
        except Exception:
            self.dataset = [{"prompt": "Test prompt."}]

        self.current_trial_stats: Dict[str, Any] = {}

    def objective(self, trial: optuna.Trial) -> float:
        # Hardware Parameters
        context_length: int = trial.suggest_categorical("context_length", [1024, 2048, 4096, 8192])
        gpu_ratio: float = trial.suggest_float("gpu_ratio", 0.0, 1.0, step=0.1)

        self.current_trial_stats = {
            "context_length": context_length,
            "gpu_ratio": gpu_ratio,
            "tps": 0.0,
            "ttft": 0.0,
            "status": "Running"
        }

        # 1. Unload existing to clear VRAM
        self.client.unload_model(self.model)
        time.sleep(2) # Give OS time to free VRAM

        # 2. Attempt to load with new hardware constraints
        success: bool = self.client.load_model(self.model, context_length, gpu_ratio)
        if not success:
            self.current_trial_stats["status"] = "OOM/Load Fail"
            return 0.0 # Heavy penalty for failing to load

        # 3. Benchmark
        prompt: str = self.dataset[0]["prompt"]
        metrics: Dict[str, Any] = self.client.generate_with_metrics(prompt, self.model)

        if not metrics.get("success", False):
            self.current_trial_stats["status"] = "Generation Fail"
            return 0.0

        tps: float = metrics["tps"]
        ttft: float = metrics["ttft"]

        self.current_trial_stats["tps"] = tps
        self.current_trial_stats["ttft"] = ttft
        self.current_trial_stats["status"] = "Success"

        # 4. Reward Function
        # We want to maximize context window, maximize TPS, and minimize TTFT.
        # Normalize context (e.g. 8192 / 8192 = 1.0)
        norm_context: float = context_length / 8192.0
        norm_tps: float = min(tps / 50.0, 1.0) # Assume 50 is peak

        # TTFT penalty: if it takes 10 seconds to start, that's bad.
        ttft_penalty: float = min(ttft / 10.0, 1.0)

        reward: float = (0.4 * norm_context) + (0.5 * norm_tps) - (0.1 * ttft_penalty)
        final_reward = max(reward, 0.0)

        trial.set_user_attr("tps", tps)
        trial.set_user_attr("ttft", ttft)
        trial.set_user_attr("status", self.current_trial_stats["status"])

        return final_reward

    def run_optimization(self, n_trials: int = 10, callback: Optional[Callable[[optuna.Trial, Dict[str, Any]], None]] = None) -> optuna.trial.FrozenTrial:
        study: optuna.Study = optuna.create_study(direction="maximize")

        # Wrap callback to pass current stats
        def optuna_callback(study: optuna.Study, trial: optuna.Trial) -> None:
            if callback:
                callback(trial, self.current_trial_stats)

        study.optimize(self.objective, n_trials=n_trials, callbacks=[optuna_callback])

        with open("optimal_settings.json", "w") as f:
            json.dump(study.best_trial.params, f, indent=4)

        return study.best_trial
