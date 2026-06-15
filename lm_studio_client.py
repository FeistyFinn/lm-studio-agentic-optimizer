import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import lmstudio
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def is_wsl() -> bool:
    """Detect whether we are running inside WSL."""
    if os.getenv("WSL_DISTRO_NAME") or os.getenv("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def get_wsl_host_ip() -> str:
    """Helper to dynamically resolve the Windows host IP from inside WSL2."""
    try:
        output = subprocess.check_output(
            ["ip", "route", "show", "default"]
        ).decode("utf-8")
        parts = output.split()
        if "via" in parts:
            idx = parts.index("via")
            return parts[idx + 1]
    except Exception:
        pass
    return "127.0.0.1"


class LMStudioHardwareClient:
    def __init__(self) -> None:
        # In WSL the LM Studio server runs on the Windows host, not localhost
        default_ip = get_wsl_host_ip() if is_wsl() else "127.0.0.1"
        self.api_host: str = os.getenv("LM_STUDIO_API_HOST", f"{default_ip}:1234")

        # 'sdk' uses the official websocket SDK (full control, including
        # gpu_ratio). 'rest' uses the HTTP API instead — it works through
        # proxies and against Bearer-token-authenticated servers, but cannot
        # set gpu_ratio (the server-side default applies).
        self.transport: str = os.getenv("LM_STUDIO_TRANSPORT", "sdk").lower()
        if self.transport not in ("sdk", "rest"):
            raise ValueError(
                f"LM_STUDIO_TRANSPORT must be 'sdk' or 'rest', "
                f"got {self.transport!r}"
            )

        self.api_key: str = os.getenv("LM_STUDIO_API_TOKEN") or os.getenv(
            "LM_STUDIO_API_KEY", "lm-studio"
        )

        self._lms: lmstudio.Client | None = None
        self._warned_gpu_ratio = False

        # Initialize standard OpenAI client for standard generation endpoint
        self.base_url: str = f"http://{self.api_host}/v1"
        self.client: OpenAI = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @property
    def lms(self) -> lmstudio.Client:
        """The websocket SDK client, created lazily so REST mode never
        opens (or fails on) a websocket connection."""
        if self._lms is None:
            self._lms = lmstudio.Client(api_host=self.api_host)
        return self._lms

    def _rest(self, method: str, path: str, payload: Any = None) -> Any:
        response = requests.request(
            method,
            f"http://{self.api_host}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def unload_model(self, model_id: str) -> bool:
        """Attempts to unload a model to free up VRAM."""
        if self.transport == "rest":
            return self._rest_unload(model_id)
        try:
            self.lms.llm.unload(model_id)
            return True
        except Exception:
            return False

    def _rest_unload(self, model_id: str) -> bool:
        try:
            listing = self._rest("GET", "/api/v1/models")
            for m in listing.get("models", []):
                if m.get("key") != model_id:
                    continue
                for inst in m.get("loaded_instances", []):
                    instance_id = inst.get("id") or inst.get("instance_id")
                    self._rest(
                        "POST",
                        "/api/v1/models/unload",
                        {"instance_id": instance_id},
                    )
            return True
        except Exception:
            return False

    def load_model(
        self, model_id: str, context_length: int, gpu_offload: float
    ) -> bool:
        """
        Dynamically loads a model with specific hardware configs.
        gpu_offload is a ratio from 0.0 to 1.0 (or "max"); REST transport
        cannot apply it and uses the server-side default.
        """
        if self.transport == "rest":
            return self._rest_load(model_id, context_length)

        gpu_offload_val: Any = "max" if gpu_offload >= 1.0 else float(gpu_offload)

        try:
            self.lms.llm.model(
                model_id,
                config=lmstudio.LlmLoadModelConfigDict(
                    contextLength=context_length,
                    gpu={"ratio": gpu_offload_val}
                )
            )
            return True
        except Exception as e:
            print(f"DEBUG SDK LOAD ERR: {e}")
            return False

    def _rest_load(self, model_id: str, context_length: int) -> bool:
        if not self._warned_gpu_ratio:
            print(
                "WARNING: REST transport cannot set gpu_ratio; the LM Studio "
                "server-side GPU configuration applies.",
                file=sys.stderr,
            )
            self._warned_gpu_ratio = True
        try:
            self._rest(
                "POST",
                "/api/v1/models/load",
                {"model": model_id, "context_length": context_length},
            )
            return True
        except Exception as e:
            print(f"DEBUG REST LOAD ERR: {e}", file=sys.stderr)
            return False

    def list_models(self) -> List[str]:
        """Returns the model keys of all downloaded LLMs."""
        if self.transport == "rest":
            listing = self._rest("GET", "/api/v1/models")
            return [
                m["key"]
                for m in listing.get("models", [])
                if m.get("type") == "llm"
            ]
        return [
            getattr(m, "model_key", str(m)) for m in self.lms.llm.list_downloaded()
        ]

    def generate_with_metrics(self, prompt: str, model: str) -> Dict[str, Any]:
        """
        Sends a prompt to LM Studio via standard OpenAI API and measures performance.
        Returns TTFT and TPS.
        """
        start_time: float = time.time()
        first_token_time: float | None = None

        try:
            # We use standard generation settings for the benchmark
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
                temperature=0.1,  # Keep temperature low for consistent benchmarking
                # Cap output so we don't wait forever. Reasoning models spend
                # tokens thinking before any visible output, so raise this
                # (e.g. BENCH_MAX_TOKENS=1024) when judging their quality.
                max_tokens=int(os.getenv("BENCH_MAX_TOKENS", "250")),
            )

            output_text: str = ""
            reasoning_text: str = ""
            completion_tokens: float = 0.0
            prompt_tokens: int = 0

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    # Reasoning models stream thinking tokens separately;
                    # they count for TTFT/TPS but are not the visible output.
                    reasoning = getattr(delta, "reasoning_content", None)
                    if content or reasoning:
                        if first_token_time is None:
                            first_token_time = time.time()
                    if content:
                        output_text += content
                    if reasoning:
                        reasoning_text += reasoning

                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    completion_tokens = chunk.usage.completion_tokens
                    prompt_tokens = chunk.usage.prompt_tokens

        except Exception as e:
            return {"error": str(e), "ttft": 0.0, "tps": 0.0, "success": False}

        end_time: float = time.time()
        ttft: float = (
            (first_token_time - start_time)
            if first_token_time
            else (end_time - start_time)
        )

        if completion_tokens == 0 and output_text:
            completion_tokens = len(output_text) / 4.0

        generation_time: float = end_time - (
            first_token_time if first_token_time else start_time
        )
        tps: float = completion_tokens / generation_time if generation_time > 0 else 0.0

        # Prompt-processing (prefill) speed: tokens ingested before the first
        # output token arrived. Only meaningful when the server reports usage.
        prefill_tps: float | None = (
            prompt_tokens / ttft if prompt_tokens and ttft > 0 else None
        )

        return {
            "output": output_text,
            "reasoning_chars": len(reasoning_text),
            "ttft": ttft,
            "tps": tps,
            "prompt_tokens": prompt_tokens,
            "prefill_tps": prefill_tps,
            "success": True
        }
