---
name: "lm-studio-agentic-benchmarker"
description: "A cross-harness skill allowing AI agents to automatically test, optimize, and benchmark local LLMs running in LM Studio."
---

# Agentic Hardware Benchmarking for LM Studio

Welcome, AI Agent! This directory contains tools to programmatically benchmark the performance of local LLMs running in LM Studio.
Your task when landing in this directory is to act as the "Intelligence" in the optimization loop to find the best hardware parameters (`context_length` and `gpu_ratio`) for a given model.

All commands below assume your shell is already inside this repository directory and that a virtual environment exists at `./.venv` (`python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt`). If you are running on a Windows host and this repository lives inside WSL, prefix commands with `wsl.exe bash -c "cd <repo-path> && <command>"`.

If the LM Studio server requires a Bearer token or sits behind a proxy that blocks websockets, set `LM_STUDIO_API_TOKEN=<token>` and `LM_STUDIO_TRANSPORT=rest`. In REST mode `--gpu_ratio` cannot be applied (the server-side GPU config is used) and `auto_optimize.py` automatically restricts its search to context length only.

## Available Tools

1. **`list_models.py`**
   - **Usage**: `./.venv/bin/python list_models.py`
   - **Purpose**: Returns a JSON list of all models currently downloaded and available in the user's LM Studio instance.
   - **Action**: Use this when the user asks you to "benchmark my models" but doesn't specify a name, or to verify the exact `model_key`.

2. **`hw_info.py`**
   - **Usage**: `./.venv/bin/python hw_info.py`
   - **Purpose**: Retrieves the Windows host's hardware specs (VRAM, RAM, GPU Name). Use this *before* running trials to estimate how much context or offload the system can handle.

3. **`run_trial.py`**
   - **Usage**: `./.venv/bin/python run_trial.py --model 'MODEL_KEY' --context_length 2048 --gpu_ratio 0.5 --preset_prompt short`
   - **Arguments**:
     - `--model`: The exact `model_key` from `list_models.py`.
     - `--context_length`: Integer (e.g. 1024, 2048, 4096, 8192).
     - `--gpu_ratio`: Float from `0.0` (0% offloaded to GPU) to `1.0` (100% offloaded to GPU).
     - `--preset_prompt`: "short", "coding", or "long". Tests context degradation and complexity.
     - `--judge`: Optional. Scores the output 1-10 with an LLM-as-a-judge (configure via `JUDGE_BASE_URL`/`JUDGE_MODEL` env vars); adds `quality_score` to the result. Note: the judge model stays loaded afterwards and will show up in the next trial's `baseline_vram_gb`.
     - `--rubric`: Grading rubric text (required when combining `--judge` with a custom `--prompt`; presets have built-in rubrics).
     - `--results PATH`: Optional. Appends the result to a JSON array file (e.g. `results.json`) so `generate_report.py` needs no manual collection.
   - **Purpose**: Loads the model into memory with the specified parameters, tests it, unloads it, and prints a JSON result with `status`, `ttft` (Time To First Token), `tps` (Tokens Per Second), `prefill_tps` (prompt-processing speed), and `peak_vram_gb` (plus `baseline_vram_gb`, the GPU usage before the load; both are `null` when telemetry is unavailable).

4. **`auto_optimize.py`**
   - **Usage**: `./.venv/bin/python auto_optimize.py --model 'MODEL_KEY' --results results.json`
   - **Arguments**: `--start_context` (default 2048), `--max_context` (default 16384), `--ratio_step` (default 0.1), plus `--preset_prompt`, `--judge`, `--results` as in `run_trial.py`.
   - **Purpose**: Deterministically finds the best `context_length`/`gpu_ratio` without you driving each trial: binary-searches the highest loadable GPU ratio, then doubles the context until OOM (backing the ratio off when needed). Prints progress to stderr, the final JSON summary to stdout, and saves the winner to `optimal_settings.json`. Use this when the user just wants "the best settings"; drive `run_trial.py` manually when you want fine-grained control.

5. **`generate_report.py`**
   - **Usage**: `./.venv/bin/python generate_report.py --input results.json --output leaderboard.md`
   - **Purpose**: Converts a JSON array of benchmark results into a clean Markdown table (includes Prefill TPS and Quality columns when present).

6. **`download_model.py`**
   - **Usage**: `./.venv/bin/python download_model.py --model 'bartowski/gemma-4-12b-GGUF'`
   - **Purpose**: Uses the LM Studio CLI to download a new model. Use this if the user approves downloading a better quantized model.

## Your Optimization Workflow

When asked to find the best configuration for a model, follow this heuristic:

1. **Establish Targets**: 
   - Check `hw_info.py` to get the GPU Name and VRAM.
   - Use your **Web Search** tool to query community benchmarks (e.g., "Gemma-4-12b RTX 3090 TPS"). Find the "Community Target Goal".
2. **Optimize**: 
   - Easiest path: run `auto_optimize.py --model 'MODEL_KEY' --results results.json` and let it find the boundary itself.
   - Manual path: start with `--context_length 2048` and `--gpu_ratio 0.5`, push `--gpu_ratio` as high as possible without hitting an OOM to maximize `tps`, then increase `--context_length` to `4096` or `8192`.
   - Add `--judge` to either path to catch configurations that are fast but produce degraded output. 
3. **Compare & Remediate**: 
   - Compare your local peak TPS to the Community Target Goal.
   - If your TPS is vastly underperforming, read the web search results to diagnose the issue (usually wrong quantization, like FP16 instead of Q4_K_M).
   - Suggest that the user downloads the correct quantized format. If the user approves, use `download_model.py` to fetch it, and re-run the benchmark!
4. **Sweep All (If Requested)**:
   - If asked to benchmark ALL models, iterate through `list_models.py`, save JSONs to `results.json`, and use `generate_report.py`.
