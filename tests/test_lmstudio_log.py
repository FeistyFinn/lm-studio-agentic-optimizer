from lmstudio_log import (
    find_main_log,
    latest_load_estimate_gb,
    parse_latest_estimate_gb,
)

LOG_TEXT = (
    "[2026-06-12 11:48:47.321] [error] [LM Studio] Model load size estimate "
    "with raw num offload layers 'max' and context length '2048':\n"
    "  Model: 6.52 GB\n"
    "  Context: 257.46 MB\n"
    "  Total: 6.77 GB\n"
    "[2026-06-12 11:48:47.322] [error] [LM Studio] Strict GPU VRAM cap is OFF\n"
    "[2026-06-12 11:52:48.600] [error] [LM Studio] Model load size estimate "
    "with raw num offload layers 'max' and context length '8192':\n"
    "  Model: 6.52 GB\n"
    "  Context: 1.01 GB\n"
    "  Total: 7.53 GB\n"
)


def test_parses_latest_estimate():
    assert parse_latest_estimate_gb(LOG_TEXT) == 7.53


def test_parses_mb_total():
    text = (
        "Model load size estimate with x:\n"
        "  Model: 300.00 MB\n"
        "  Total: 512.00 MB\n"
    )
    assert parse_latest_estimate_gb(text) == 0.5


def test_no_estimate_returns_none():
    assert parse_latest_estimate_gb("nothing relevant here") is None


def test_latest_load_estimate_reads_file(tmp_path, monkeypatch):
    log = tmp_path / "main.log"
    log.write_text(LOG_TEXT)
    monkeypatch.setenv("LM_STUDIO_LOG_PATH", str(log))

    assert latest_load_estimate_gb() == 7.53


def test_missing_env_path_returns_none(monkeypatch):
    monkeypatch.setenv("LM_STUDIO_LOG_PATH", "/nonexistent/main.log")

    assert find_main_log() is None
    assert latest_load_estimate_gb() is None
