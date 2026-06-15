import argparse
import json
import os

import optuna

# Suppress verbose logs for clean loop
optuna.logging.set_verbosity(optuna.logging.ERROR)

from lm_studio_client import LMStudioHardwareClient  # noqa: E402
from optimizer_agent import HardwareOptimizerAgent  # noqa: E402


def discover_models():
    return LMStudioHardwareClient().list_models()


def main():
    parser = argparse.ArgumentParser(
        description="Run the Optuna sweep across multiple models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys to benchmark (default: all downloaded models)",
    )
    parser.add_argument(
        "--trials", type=int, default=8, help="Optimization trials per model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Path for the results JSON",
    )
    args = parser.parse_args()

    models_to_bench = args.models or discover_models()
    if not models_to_bench:
        print("No models found to benchmark.")
        return

    results = {}

    print(
        "Starting autonomous benchmark sequence for "
        f"{len(models_to_bench)} models...\n"
    )

    for model in models_to_bench:
        print(f"[{model}] Benchmarking...")
        os.environ["TARGET_MODEL"] = model

        agent = HardwareOptimizerAgent(dataset_path="dataset.json")

        best_trial = agent.run_optimization(n_trials=args.trials)

        if best_trial.value == 0.0:
            print(f"[{model}] Failed (All trials OOM/Load Fail)\n")
            results[model] = {
                "status": "Failed",
                "reason": "OOM or Load Failure across all trials",
            }
        else:
            tps = best_trial.user_attrs.get("tps", 0.0)
            ttft = best_trial.user_attrs.get("ttft", 0.0)
            print(f"[{model}] Success! Best TPS: {tps:.2f}, TTFT: {ttft:.2f}s")
            print(f"[{model}] Params: {best_trial.params}\n")

            results[model] = {
                "status": "Success",
                "context_length": best_trial.params.get("context_length"),
                "gpu_ratio": round(best_trial.params.get("gpu_ratio", 0.0), 2),
                "tps": round(tps, 2),
                "ttft": round(ttft, 2),
                "reward_score": round(best_trial.value, 4),
            }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Benchmark run complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
