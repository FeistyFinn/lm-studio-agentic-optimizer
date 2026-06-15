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


def make_chunk(content=None, reasoning=None, usage=None):
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(delta=MagicMock(content=content, reasoning_content=reasoning))
    ]
    chunk.usage = usage
    return chunk


def test_generate_with_metrics(client):
    mock_chunk_1 = make_chunk(content="Hello ")
    mock_chunk_2 = make_chunk(
        content="World",
        usage=MagicMock(completion_tokens=2, prompt_tokens=10),
    )

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
    assert result["reasoning_chars"] == 0


def test_generate_with_metrics_reasoning_only(client):
    # Reasoning models can spend the whole token budget thinking: TTFT must
    # count the first reasoning token, and output stays empty.
    chunks = [
        make_chunk(reasoning="thinking..."),
        make_chunk(usage=MagicMock(completion_tokens=250, prompt_tokens=27)),
    ]
    client.client.chat.completions.create.return_value = chunks

    with patch("lm_studio_client.time.time", side_effect=[100.0, 100.5, 110.5]):
        result = client.generate_with_metrics("prompt", "test-model")

    assert result["output"] == ""
    assert result["reasoning_chars"] == len("thinking...")
    assert result["ttft"] == 0.5  # first reasoning token, not end of stream
    assert result["tps"] == 25.0  # 250 tokens / 10s


def test_generate_with_metrics_error(client):
    client.client.chat.completions.create.side_effect = RuntimeError("conn refused")

    result = client.generate_with_metrics("prompt", "test-model")

    assert result["success"] is False
    assert "conn refused" in result["error"]


@pytest.fixture
def rest_client(monkeypatch):
    monkeypatch.setenv("LM_STUDIO_TRANSPORT", "rest")
    monkeypatch.setenv("LM_STUDIO_API_TOKEN", "secret-token")
    with patch("lm_studio_client.OpenAI"), patch(
        "lm_studio_client.is_wsl", return_value=False
    ), patch("lm_studio_client.requests.request") as mock_request:
        client = LMStudioHardwareClient()
        client._mock_request = mock_request
        yield client


def rest_response(payload, status=200):
    response = MagicMock()
    response.json.return_value = payload
    response.status_code = status
    if status >= 400:
        response.raise_for_status.side_effect = RuntimeError(f"HTTP {status}")
    return response


def test_rest_transport_invalid_value(monkeypatch):
    monkeypatch.setenv("LM_STUDIO_TRANSPORT", "carrier-pigeon")
    with patch("lm_studio_client.OpenAI"), patch(
        "lm_studio_client.is_wsl", return_value=False
    ):
        with pytest.raises(ValueError):
            LMStudioHardwareClient()


def test_rest_token_used_for_openai_key(rest_client):
    assert rest_client.api_key == "secret-token"


def test_rest_load_model(rest_client, capsys):
    rest_client._mock_request.return_value = rest_response({"status": "loaded"})

    assert rest_client.load_model("m", 4096, 0.5) is True

    args, kwargs = rest_client._mock_request.call_args
    assert args == ("POST", "http://127.0.0.1:1234/api/v1/models/load")
    assert kwargs["json"] == {"model": "m", "context_length": 4096}
    assert kwargs["headers"]["Authorization"] == "Bearer secret-token"
    assert "cannot set gpu_ratio" in capsys.readouterr().err


def test_rest_load_model_failure(rest_client):
    rest_client._mock_request.return_value = rest_response({}, status=500)

    assert rest_client.load_model("m", 32768, 1.0) is False


def test_rest_unload_unloads_all_instances(rest_client):
    rest_client._mock_request.side_effect = [
        rest_response(
            {
                "models": [
                    {
                        "key": "m",
                        "loaded_instances": [{"id": "i1"}, {"id": "i2"}],
                    },
                    {"key": "other", "loaded_instances": [{"id": "i3"}]},
                ]
            }
        ),
        rest_response({"instance_id": "i1"}),
        rest_response({"instance_id": "i2"}),
    ]

    assert rest_client.unload_model("m") is True

    unload_calls = [
        c for c in rest_client._mock_request.call_args_list
        if c.args[1].endswith("/unload")
    ]
    assert [c.kwargs["json"]["instance_id"] for c in unload_calls] == ["i1", "i2"]


def test_rest_list_models_filters_llms(rest_client):
    rest_client._mock_request.return_value = rest_response(
        {
            "models": [
                {"key": "llm-1", "type": "llm"},
                {"key": "embed-1", "type": "embeddings"},
                {"key": "llm-2", "type": "llm"},
            ]
        }
    )

    assert rest_client.list_models() == ["llm-1", "llm-2"]


def test_sdk_list_models(client):
    model = MagicMock()
    model.model_key = "sdk-model"
    client.lms.llm.list_downloaded.return_value = [model]

    assert client.list_models() == ["sdk-model"]
