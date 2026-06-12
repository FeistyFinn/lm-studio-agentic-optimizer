import json
import re
import subprocess
import threading
import time

from lm_studio_client import LMStudioHardwareClient
from lmstudio_log import latest_load_estimate_gb

PRESET_PROMPTS = {
    "short": "Write a python script to calculate the fibonacci sequence.",
    "coding": (
        "Write a fully functional, production-ready REST API in Python using "
        "FastAPI with SQLAlchemy integration, Pydantic models, and JWT "
        "authentication for a blog platform."
    ),
    "long": (
        "Please summarize the history of the Roman Empire, detailing the rise "
        "of Julius Caesar, the transition from Republic to Empire, the Pax "
        "Romana, the crisis of the third century, and the eventual fall of the "
        "Western Roman Empire. Include a section on the Byzantine Empire's "
        "survival. "
    )
    * 10,
}

# Benchmark generations are capped at 250 tokens, so rubrics ask the judge to
# grade the quality of what is present rather than penalizing truncation.
PRESET_RUBRICS = {
    "short": (
        "The response must contain a syntactically valid Python script that "
        "computes the Fibonacci sequence. Score highly for correct, clean, "
        "runnable code; score low for broken syntax, incoherence, or content "
        "unrelated to the task. The output may be truncated mid-script; judge "
        "the quality of what is present."
    ),
    "coding": (
        "The response must be the start of a coherent FastAPI REST API "
        "implementation featuring SQLAlchemy integration, Pydantic models, and "
        "JWT authentication. Score on correctness and structure of the code "
        "present; score low for incoherent or off-topic output. The output is "
        "truncated at 250 tokens; do not penalize incompleteness."
    ),
    "long": (
        "The response must be an accurate, coherent summary of Roman Empire "
        "history touching on: Julius Caesar's rise, the Republic-to-Empire "
        "transition, the Pax Romana, the crisis of the third century, the fall "
        "of the Western Empire, and the Byzantine Empire's survival. Score on "
        "accuracy and coherence of what is present; the output is truncated at "
        "250 tokens, so do not penalize incompleteness."
    ),
}


class TelemetryMonitor:
    """Polls whole-GPU VRAM usage via nvidia-smi on the Windows host.

    Note: this measures total GPU memory in use (including other processes),
    not just the model being benchmarked. ``baseline_vram_gb`` captures usage
    before the model load so consumers can compute the delta.
    """

    # Sampler commands tried in order; the first that works is kept.
    SAMPLER_COMMANDS = (
        ["nvidia-smi", "-q", "-d", "MEMORY"],
        ["powershell.exe", "-c", "nvidia-smi -q -d MEMORY"],
    )

    def __init__(self):
        self.peak_vram_gb = None
        self.baseline_vram_gb = None
        self.running = False
        self.thread = None
        self._working_command = None

    @staticmethod
    def _query_used_gb(command):
        smi_out = subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        )
        match = re.search(
            r"FB Memory Usage.*?\n.*?Used\s*:\s*(\d+)\s*MiB", smi_out, re.DOTALL
        )
        if match:
            return round(int(match.group(1)) / 1024.0, 1)
        return None

    def _sample_vram_gb(self):
        commands = (
            [self._working_command]
            if self._working_command is not None
            else self.SAMPLER_COMMANDS
        )
        for command in commands:
            try:
                used_gb = self._query_used_gb(command)
            except Exception:
                continue
            if used_gb is not None:
                self._working_command = command
                return used_gb
        return None

    def _monitor_loop(self):
        while self.running:
            used_gb = self._sample_vram_gb()
            if used_gb is not None:
                if self.baseline_vram_gb is None:
                    self.baseline_vram_gb = used_gb
                if self.peak_vram_gb is None or used_gb > self.peak_vram_gb:
                    self.peak_vram_gb = used_gb
            time.sleep(0.5)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stops monitoring. Returns peak VRAM in GB, or None if unknown."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.peak_vram_gb


def append_result(path: str, result: dict) -> None:
    """Appends a trial result to a JSON array file, creating it if missing."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(result)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_single_trial(
    model: str,
    context_length: int,
    gpu_ratio: float,
    prompt: str,
    judge=None,
    rubric: str | None = None,
    client: LMStudioHardwareClient | None = None,
) -> dict:
    """Runs one load/benchmark/unload cycle and returns the result dict.

    ``judge`` is an optional LLMEvaluator-like object; when provided together
    with ``rubric``, the generated output is scored and ``quality_score`` /
    ``judge_reasoning`` are added to the result.
    """
    client = client or LMStudioHardwareClient()

    base = {
        "model": model,
        "context_length": context_length,
        "gpu_ratio": gpu_ratio,
        "transport": getattr(client, "transport", "sdk"),
    }

    # 1. Unload model to clear VRAM (so we start fresh)
    client.unload_model(model)
    time.sleep(1)  # Let VRAM clear completely

    telemetry = TelemetryMonitor()
    telemetry.start()

    # 2. Attempt to load with new hardware constraints
    success = client.load_model(model, context_length, gpu_ratio)

    if not success:
        peak_vram = telemetry.stop()
        return {
            **base,
            "status": "OOM/Load Fail",
            "ttft": 0.0,
            "tps": 0.0,
            "baseline_vram_gb": telemetry.baseline_vram_gb,
            "peak_vram_gb": peak_vram,
        }

    # 3. Benchmark
    metrics = client.generate_with_metrics(prompt, model)
    peak_vram = telemetry.stop()

    if not metrics.get("success", False):
        return {
            **base,
            "status": "Generation Fail",
            "error": metrics.get("error", "Unknown error"),
            "ttft": 0.0,
            "tps": 0.0,
            "baseline_vram_gb": telemetry.baseline_vram_gb,
            "peak_vram_gb": peak_vram,
        }

    result = {
        **base,
        "status": "Success",
        "ttft": metrics["ttft"],
        "tps": metrics["tps"],
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "prefill_tps": metrics.get("prefill_tps"),
        "reasoning_chars": metrics.get("reasoning_chars", 0),
        "baseline_vram_gb": telemetry.baseline_vram_gb,
        "peak_vram_gb": peak_vram,
        "output_preview": (
            metrics["output"][:100] + "..." if metrics.get("output") else ""
        ),
    }

    # When no sampler can see the benchmark GPU (e.g. AMD cards), fall back
    # to LM Studio's own load-size estimate from its application log.
    if peak_vram is None:
        estimate = latest_load_estimate_gb()
        if estimate is not None:
            result["vram_estimate_gb"] = estimate

    # 4. Optional quality scoring (runs after telemetry so the judge model's
    # VRAM usage does not pollute this trial's peak, and before the unload
    # below so self-judging reuses the already-loaded model)
    if judge is not None and rubric:
        output = metrics.get("output", "")
        if not output and result["reasoning_chars"] > 0:
            # The model spent the whole token budget thinking — a benchmark
            # configuration artifact, not a quality signal. Raise
            # BENCH_MAX_TOKENS to give reasoning models room to answer.
            result["quality_score"] = None
            result["judge_reasoning"] = (
                "skipped: reasoning consumed the token budget "
                f"({result['reasoning_chars']} reasoning chars, no output)"
            )
        else:
            evaluation = judge.evaluate(prompt, rubric, output)
            result["quality_score"] = evaluation.get("score")
            reasoning = evaluation.get("reasoning") or ""
            result["judge_reasoning"] = reasoning[:300]

    # 5. Leave VRAM as we found it — otherwise the tested model stays
    # resident and skews every subsequent trial of a *different* model
    client.unload_model(model)

    return result
