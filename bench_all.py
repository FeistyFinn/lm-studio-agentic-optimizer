import os
import json
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR) # Suppress verbose logs for clean loop
from optimizer_agent import HardwareOptimizerAgent

models_to_bench = [
    "google/gemma-4-26b-a4b-qat",
    "google/gemma-4-31b-qat",
    "google/gemma-4-12b",
    "google/gemma-4-26b-a4b",
    "google/gemma-4-31b"
]

results = {}

print("Starting autonomous benchmark sequence for Gemma 4 models...\n")

for model in models_to_bench:
    print(f"[{model}] Benchmarking...")
    os.environ["TARGET_MODEL"] = model

    agent = HardwareOptimizerAgent(dataset_path="dataset.json")

    # Run 5 trials per model for speed, but enough to probe the bounds
    best_trial = agent.run_optimization(n_trials=8)

    if best_trial.value == 0.0:
        print(f"[{model}] ❌ Failed (All trials OOM/Load Fail)\n")
        results[model] = {
            "status": "Failed",
            "reason": "OOM or Load Failure across all trials"
        }
    else:
        tps = best_trial.user_attrs.get("tps", 0.0)
        ttft = best_trial.user_attrs.get("ttft", 0.0)
        print(f"[{model}] ✅ Success! Best TPS: {tps:.2f}, TTFT: {ttft:.2f}s")
        print(f"[{model}] Params: {best_trial.params}\n")

        results[model] = {
            "status": "Success",
            "context_length": best_trial.params.get("context_length"),
            "gpu_ratio": round(best_trial.params.get("gpu_ratio", 0.0), 2),
            "tps": round(tps, 2),
            "ttft": round(ttft, 2),
            "reward_score": round(best_trial.value, 4)
        }

with open("gemma_4_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Benchmark run complete. Results saved to gemma_4_benchmark_results.json")
