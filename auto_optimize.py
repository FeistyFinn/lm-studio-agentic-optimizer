import argparse
import json
import sys

from trial_runner import (
    PRESET_PROMPTS,
    PRESET_RUBRICS,
    append_result,
    run_single_trial,
)

# How many ratio step-downs to attempt when a context doubling hits OOM
# before giving up on larger contexts.
MAX_RATIO_BACKOFF_STEPS = 2


def _ratio_grid(step: float) -> list[float]:
    grid = []
    r = 0.0
    while r < 1.0:
        grid.append(round(r, 2))
        r += step
    grid.append(1.0)
    return grid


def find_max_loadable_ratio(run, ratios):
    """Binary-searches the highest loadable gpu_ratio on an ascending grid.

    ``run(ratio)`` must return a trial result dict. Assumes loadability is
    monotone: if a ratio OOMs, every higher ratio also OOMs.
    Returns (ratio, result) or (None, None) if nothing loads.
    """
    hi = len(ratios) - 1

    # Optimistic fast path: full offload usually fits
    result = run(ratios[hi])
    if result["status"] == "Success":
        return ratios[hi], result
    hi -= 1

    lo = 0
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        result = run(ratios[mid])
        if result["status"] == "Success":
            best = (ratios[mid], result)
            lo = mid + 1
        else:
            hi = mid - 1
    return best if best else (None, None)


def optimize(
    runner,
    start_context: int = 2048,
    max_context: int = 16384,
    ratio_step: float = 0.1,
    log=print,
):
    """Deterministic search for the best (context_length, gpu_ratio) config.

    ``runner(context_length, gpu_ratio)`` runs one trial and returns its
    result dict. Returns (best_result_or_None, all_trial_results).

    Strategy: find the max loadable gpu_ratio at start_context via binary
    search, then double the context until OOM or max_context, backing the
    ratio off by up to MAX_RATIO_BACKOFF_STEPS steps when a doubling fails.
    The best config is the successful trial with the highest TPS, breaking
    ties toward larger context.
    """
    trials = []

    def run_at(ctx, ratio):
        log(f"Trial: context_length={ctx}, gpu_ratio={ratio}")
        result = runner(ctx, ratio)
        trials.append(result)
        log(f"  -> {result['status']} (tps={result.get('tps', 0.0):.2f})")
        return result

    ratios = _ratio_grid(ratio_step)
    log(f"Phase 1: finding max loadable gpu_ratio at context {start_context}")
    ratio, _ = find_max_loadable_ratio(
        lambda r: run_at(start_context, r), ratios
    )
    if ratio is None:
        log("No gpu_ratio loads at the starting context; aborting.")
        return None, trials

    log(f"Phase 2: growing context from {start_context} (ratio {ratio})")
    ctx = start_context
    while ctx * 2 <= max_context:
        next_ctx = ctx * 2
        result = run_at(next_ctx, ratio)
        if result["status"] == "Success":
            ctx = next_ctx
            continue

        backed_off = False
        r = ratio
        for _ in range(MAX_RATIO_BACKOFF_STEPS):
            r = round(r - ratio_step, 2)
            if r < 0:
                break
            result = run_at(next_ctx, r)
            if result["status"] == "Success":
                ratio, ctx = r, next_ctx
                backed_off = True
                break
        if not backed_off:
            break

    successes = [t for t in trials if t["status"] == "Success"]
    if not successes:
        return None, trials
    best = max(successes, key=lambda t: (t["tps"], t["context_length"]))
    return best, trials


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically find the best context_length/gpu_ratio for a "
            "model without OOM."
        )
    )
    parser.add_argument("--model", type=str, required=True, help="Model ID to test")
    parser.add_argument(
        "--start_context", type=int, default=2048, help="Initial context length"
    )
    parser.add_argument(
        "--max_context", type=int, default=16384, help="Context length ceiling"
    )
    parser.add_argument(
        "--ratio_step", type=float, default=0.1, help="GPU ratio grid granularity"
    )
    parser.add_argument(
        "--preset_prompt",
        type=str,
        choices=sorted(PRESET_PROMPTS),
        default="short",
        help="Benchmark prompt complexity",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Score output quality with an LLM-as-a-judge (configure via JUDGE_* env)",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Append every trial to this JSON array file",
    )

    args = parser.parse_args()

    prompt = PRESET_PROMPTS[args.preset_prompt]
    rubric = PRESET_RUBRICS[args.preset_prompt]

    judge = None
    if args.judge:
        from evaluator import LLMEvaluator

        judge = LLMEvaluator()

    def runner(context_length, gpu_ratio):
        result = run_single_trial(
            model=args.model,
            context_length=context_length,
            gpu_ratio=gpu_ratio,
            prompt=prompt,
            judge=judge,
            rubric=rubric,
        )
        if args.results:
            append_result(args.results, result)
        return result

    def log(msg):
        print(msg, file=sys.stderr)

    best, trials = optimize(
        runner,
        start_context=args.start_context,
        max_context=args.max_context,
        ratio_step=args.ratio_step,
        log=log,
    )

    if best is None:
        print(
            json.dumps(
                {
                    "status": "Failed",
                    "reason": "No configuration loaded successfully",
                    "trials_run": len(trials),
                }
            )
        )
        sys.exit(1)

    optimal = {
        "context_length": best["context_length"],
        "gpu_ratio": best["gpu_ratio"],
    }
    with open("optimal_settings.json", "w") as f:
        json.dump(optimal, f, indent=4)

    print(
        json.dumps(
            {
                "status": "Success",
                "best": best,
                "trials_run": len(trials),
                "saved_to": "optimal_settings.json",
            }
        )
    )


if __name__ == "__main__":
    main()
