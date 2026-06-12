"""Reads VRAM load-size estimates from LM Studio's own application log.

When the benchmark GPU is invisible to nvidia-smi (AMD/Intel cards, or
remote setups), LM Studio's main.log still records a size estimate for every
model load, e.g.:

    [...] Model load size estimate with raw num offload layers 'max' ...:
      Model: 6.52 GB
      Context: 257.46 MB
      Total: 6.77 GB
"""

import glob
import os
import re

ESTIMATE_RE = re.compile(
    r"Model load size estimate[^\n]*\n(?:[^\n]*\n){0,4}?\s*Total:\s*"
    r"([\d.]+)\s*(GB|MB)"
)

TAIL_BYTES = 256 * 1024


def find_main_log() -> str | None:
    env_path = os.getenv("LM_STUDIO_LOG_PATH")
    if env_path:
        return env_path if os.path.exists(env_path) else None
    candidates = glob.glob("/mnt/c/Users/*/AppData/Roaming/LM Studio/logs/main.log")
    candidates.append(os.path.expanduser("~/.config/LM Studio/logs/main.log"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def parse_latest_estimate_gb(text: str) -> float | None:
    matches = ESTIMATE_RE.findall(text)
    if not matches:
        return None
    value, unit = matches[-1]
    gb = float(value) / 1024.0 if unit == "MB" else float(value)
    return round(gb, 2)


def latest_load_estimate_gb(log_path: str | None = None) -> float | None:
    """Returns LM Studio's most recent model-load size estimate in GB."""
    path = log_path or find_main_log()
    if path is None:
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    return parse_latest_estimate_gb(text)
