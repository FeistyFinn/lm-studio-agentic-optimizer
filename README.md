# Agentic LM Studio Benchmark Harness

This repository contains tools to programmatically discover, test, and optimize the hardware parameters for local Large Language Models running in [LM Studio](https://lmstudio.ai/).

While there are legacy Optuna-based programmatic sweep tools (`main.py`), the primary focus of this repository is to act as an **Agentic Tool Suite**. By pointing an autonomous coding agent (such as Antigravity, Claude Code, or Hermes) at this directory, the agent can use its own intelligence to benchmark your models and optimize your hardware settings.

## Features

- **Agentic Native**: Fully encapsulated skills via `.claudeproject` and `SKILL.md` to teach AI agents how to optimize your system.
- **Deep Telemetry**: Real-time VRAM tracking using `nvidia-smi` to ensure models fit within memory limits.
- **Auto-Remediation**: Agents can search the web for community benchmark targets, identify if your model is underperforming (e.g., wrong quantization), and use the LM Studio CLI (`lms`) to automatically download the correct model to quadruple your tokens per second!
- **Dynamic Leaderboards**: Generate pristine Markdown tables of your hardware's benchmark results.

## Requirements

- LM Studio running locally with the CLI (`lms`) installed.
- Python 3.10+
- `pip install -r requirements.txt`
- Windows Subsystem for Linux (WSL) is supported! The tools automatically detect WSL environments and bridge connections to the Windows host.

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
4. Run iterative loops of `run_trial.py` to push your `gpu_ratio` and `context_length` to their absolute maximum limits without hitting an Out-of-Memory error.
5. If the maximum TPS falls short of the community target, the Agent will ask you to let it download a better model (like a Q4_K_M GGUF) via `download_model.py`.
6. Finally, the Agent will use `generate_report.py` to build a clean Markdown leaderboard.
