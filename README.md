# Agentic LM Studio Benchmark Harness

This repository contains tools to programmatically discover, test, and optimize the hardware parameters for local Large Language Models running in [LM Studio](https://lmstudio.ai/).

While there are legacy Optuna-based programmatic sweep tools (`main.py`), the primary focus of this repository is to act as an **Agentic Tool Suite**. By pointing an autonomous coding agent (such as Antigravity, Claude Code, or Hermes) at this directory, the agent can use its own intelligence to benchmark your models and optimize your hardware settings.

A sample benchmark sweep against three Gemma/Qwen models on an AMD WX 7100 lives in [BENCHMARKS.md](BENCHMARKS.md).

## Features

- **Agentic Native**: Fully encapsulated skills via `.claudeproject` and `SKILL.md` to teach AI agents how to optimize your system.
- **Deterministic Auto-Optimizer**: `auto_optimize.py` binary-searches the highest loadable GPU offload ratio and doubles the context length until OOM — no agent babysitting required.
- **Quality-Aware Benchmarks**: Optional LLM-as-a-judge scoring (`--judge`) catches configurations that are fast but produce degraded output, and prefill TPS measures prompt-processing speed alongside generation TPS.
- **Deep Telemetry**: Real-time VRAM tracking using `nvidia-smi` to ensure models fit within memory limits, including the pre-load baseline.
- **Auto-Remediation**: Agents can search the web for community benchmark targets, identify if your model is underperforming (e.g., wrong quantization), and use the LM Studio CLI (`lms`) to automatically download the correct model to quadruple your tokens per second!
- **Dynamic Leaderboards**: Generate pristine Markdown tables of your hardware's benchmark results.

## Requirements

- LM Studio running locally with the CLI (`lms`) installed.
- Python 3.10+
- `pip install -r requirements.txt`
- Windows Subsystem for Linux (WSL) is supported! The tools automatically detect WSL environments and bridge connections to the Windows host.

### Environment variables

| Variable | Purpose |
|---|---|
| `LM_STUDIO_API_HOST` | Override the auto-detected `host:port` (default `127.0.0.1:1234`, WSL host IP inside WSL) |
| `LM_STUDIO_API_TOKEN` | Bearer token for authenticated LM Studio servers (used for both transports) |
| `LM_STUDIO_TRANSPORT` | `sdk` (default, websocket — full control incl. `gpu_ratio`) or `rest` (HTTP — works through proxies and Bearer auth, but `gpu_ratio` falls back to the server-side default) |
| `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL` | LLM-as-a-judge endpoint for `--judge` quality scoring |
| `TPS_REWARD_CEILING` | TPS at which the Optuna reward saturates (default 50) |
| `LM_STUDIO_LOG_PATH` | Path to LM Studio's `main.log` for VRAM load-size estimates (auto-discovered on WSL/Linux) |

## Agent Workflow

To use this directory, simply open your favorite autonomous AI Agent inside this directory.

```bash
# Example with Claude Code
claude

# Example with Antigravity
ag
```

Ask the agent: *"Benchmark my models and find the optimal settings."*

The Agent will autonomously:
1. Call `hw_info.py` to check your available VRAM.
2. Search the web for community benchmarks of your GPU + Model combo.
3. Call `list_models.py` to see what models you have loaded.
4. Run `auto_optimize.py` (or iterative loops of `run_trial.py`) to push your `gpu_ratio` and `context_length` to their absolute maximum limits without hitting an Out-of-Memory error, optionally scoring output quality with `--judge`.
5. If the maximum TPS falls short of the community target, the Agent will ask you to let it download a better model (like a Q4_K_M GGUF) via `download_model.py`.
6. Finally, the Agent will use `generate_report.py` to build a clean Markdown leaderboard.
