import argparse
import json
import sys

from trial_runner import (
    PRESET_PROMPTS,
    PRESET_RUBRICS,
    append_result,
    run_single_trial,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run a single LM Studio hardware benchmark trial."
    )
    parser.add_argument("--model", type=str, required=True, help="Model ID to test")
    parser.add_argument(
        "--context_length", type=int, required=True, help="Context length to load"
    )
    parser.add_argument(
        "--gpu_ratio",
        type=float,
        required=True,
        help="GPU offload ratio (0.0 to 1.0)",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Custom test prompt")
    parser.add_argument(
        "--preset_prompt",
        type=str,
        choices=sorted(PRESET_PROMPTS),
        default="short",
        help="Use a predefined prompt complexity",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Score output quality with an LLM-as-a-judge (configure via JUDGE_* env)",
    )
    parser.add_argument(
        "--rubric",
        type=str,
        default=None,
        help="Grading rubric for --judge (required with a custom --prompt)",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Append the result to this JSON array file (e.g. results.json)",
    )

    args = parser.parse_args()

    if args.prompt:
        final_prompt = args.prompt
        rubric = args.rubric
    else:
        final_prompt = PRESET_PROMPTS[args.preset_prompt]
        rubric = args.rubric or PRESET_RUBRICS[args.preset_prompt]

    judge = None
    if args.judge:
        if not rubric:
            print(
                json.dumps(
                    {
                        "status": "Config Error",
                        "error": "--judge with a custom --prompt requires --rubric",
                    }
                )
            )
            sys.exit(1)
        from evaluator import LLMEvaluator

        judge = LLMEvaluator()

    result = run_single_trial(
        model=args.model,
        context_length=args.context_length,
        gpu_ratio=args.gpu_ratio,
        prompt=final_prompt,
        judge=judge,
        rubric=rubric,
    )

    if args.results:
        append_result(args.results, result)

    print(json.dumps(result))

    if result["status"] != "Success":
        sys.exit(1)


if __name__ == "__main__":
    main()
