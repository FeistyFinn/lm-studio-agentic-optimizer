from unittest.mock import MagicMock, mock_open, patch

import pytest

from lm_studio_client import LMStudioHardwareClient, is_wsl


@pytest.fixture
def client():
    # Patch both SDK clients so tests never touch the network or env proxies
    with patch("lm_studio_client.lmstudio.Client"), patch("lm_studio_client.OpenAI"):
        with patch("lm_studio_client.is_wsl", return_value=False):
            yield LMStudioHardwareClient()


def test_default_host_outside_wsl(client):
    assert client.api_host == "127.0.0.1:1234"


def test_is_wsl_detects_microsoft_kernel(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    data = "Linux version 6.6.0-microsoft-standard-WSL2"
    with patch("builtins.open", mock_open(read_data=data)):
        assert is_wsl() is True


def test_is_wsl_false_on_plain_linux(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    data = "Linux version 6.6.0-generic (gcc ...)"
    with patch("builtins.open", mock_open(read_data=data)):
        assert is_wsl() is False


def test_load_model_success(client):
    client.lms.llm.model.return_value = MagicMock()

    success = client.load_model("test-model", 2048, 0.5)

    assert success is True
    config = client.lms.llm.model.call_args.kwargs["config"]
    assert config["contextLength"] == 2048
    assert config["gpu"] == {"ratio": 0.5}


def test_load_model_full_offload_uses_max(client):
    client.load_model("test-model", 4096, 1.0)

    config = client.lms.llm.model.call_args.kwargs["config"]
    assert config["gpu"] == {"ratio": "max"}


def test_load_model_failure(client):
    client.lms.llm.model.side_effect = RuntimeError("Out of memory")

    success = client.load_model("test-model", 32768, 1.0)

    assert success is False


def test_unload_model(client):
    success = client.unload_model("test-model")

    assert success is True
    client.lms.llm.unload.assert_called_once_with("test-model")


def test_unload_model_failure(client):
    client.lms.llm.unload.side_effect = RuntimeError("Model not loaded")

    assert client.unload_model("test-model") is False


def test_generate_with_metrics(client):
    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
    mock_chunk_1.usage = None

    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices = [MagicMock(delta=MagicMock(content="World"))]
    mock_chunk_2.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    client.client.chat.completions.create.return_value = [mock_chunk_1, mock_chunk_2]

    # time.time() is called exactly three times: start, first token, end
    with patch("lm_studio_client.time.time", side_effect=[100.0, 100.5, 102.5]):
        result = client.generate_with_metrics("Say Hello World", "test-model")

    assert result["success"] is True
    assert result["output"] == "Hello World"
    assert result["ttft"] == 0.5
    assert result["tps"] == 1.0  # 2 tokens / 2.0s generation time
    assert result["prompt_tokens"] == 10
    assert result["prefill_tps"] == 20.0  # 10 prompt tokens / 0.5s TTFT


def test_generate_with_metrics_error(client):
    client.client.chat.completions.create.side_effect = RuntimeError("conn refused")

    result = client.generate_with_metrics("prompt", "test-model")

    assert result["success"] is False
    assert "conn refused" in result["error"]
