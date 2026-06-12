import argparse
import json
import re
import subprocess
import sys
import threading
import time

from lm_studio_client import LMStudioHardwareClient


class TelemetryMonitor:
    """Polls whole-GPU VRAM usage via nvidia-smi on the Windows host.

    Note: this measures total GPU memory in use (including other processes),
    not just the model being benchmarked. ``baseline_vram_gb`` captures usage
    before the model load so consumers can compute the delta.
    """

    def __init__(self):
        self.peak_vram_gb = None
        self.baseline_vram_gb = None
        self.running = False
        self.thread = None

    @staticmethod
    def _sample_vram_gb():
        try:
            smi_out = subprocess.check_output(
                ["powershell.exe", "-c", "nvidia-smi -q -d MEMORY"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return None
        match = re.search(
            r"FB Memory Usage.*?\n.*?Used\s*:\s*(\d+)\s*MiB", smi_out, re.DOTALL
        )
        if match:
            return round(int(match.group(1)) / 1024.0, 1)
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
        choices=["short", "coding", "long"],
        default="short",
        help="Use a predefined prompt complexity",
    )

    args = parser.parse_args()

    presets = {
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

    final_prompt = args.prompt if args.prompt else presets[args.preset_prompt]

    client = LMStudioHardwareClient()

    # 1. Unload model to clear VRAM (so we start fresh)
    client.unload_model(args.model)
    time.sleep(1)  # Let VRAM clear completely

    # Start Telemetry
    telemetry = TelemetryMonitor()
    telemetry.start()

    # 2. Attempt to load with new hardware constraints
    success = client.load_model(args.model, args.context_length, args.gpu_ratio)

    if not success:
        peak_vram = telemetry.stop()
        result = {
            "model": args.model,
            "context_length": args.context_length,
            "gpu_ratio": args.gpu_ratio,
            "status": "OOM/Load Fail",
            "ttft": 0.0,
            "tps": 0.0,
            "baseline_vram_gb": telemetry.baseline_vram_gb,
            "peak_vram_gb": peak_vram,
        }
        print(json.dumps(result))
        sys.exit(1)

    # 3. Benchmark
    metrics = client.generate_with_metrics(final_prompt, args.model)
    peak_vram = telemetry.stop()

    if not metrics.get("success", False):
        result = {
            "model": args.model,
            "context_length": args.context_length,
            "gpu_ratio": args.gpu_ratio,
            "status": "Generation Fail",
            "error": metrics.get("error", "Unknown error"),
            "ttft": 0.0,
            "tps": 0.0,
            "baseline_vram_gb": telemetry.baseline_vram_gb,
            "peak_vram_gb": peak_vram,
        }
        print(json.dumps(result))
        sys.exit(1)

    result = {
        "model": args.model,
        "context_length": args.context_length,
        "gpu_ratio": args.gpu_ratio,
        "status": "Success",
        "ttft": metrics["ttft"],
        "tps": metrics["tps"],
        "baseline_vram_gb": telemetry.baseline_vram_gb,
        "peak_vram_gb": peak_vram,
        "output_preview": (
            metrics["output"][:100] + "..." if metrics.get("output") else ""
        ),
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
