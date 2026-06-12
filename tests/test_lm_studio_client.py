import pytest
import responses
from unittest.mock import patch, MagicMock
from lm_studio_client import LMStudioHardwareClient

@pytest.fixture
def client():
    return LMStudioHardwareClient()

@responses.activate
def test_load_model_success(client):
    # Mock the LM Studio management API endpoint
    url = f"{client.management_url}/models/load"
    responses.add(
        responses.POST,
        url,
        json={"status": "loaded"},
        status=200
    )

    success = client.load_model("test-model", 2048, 0.5)
    assert success is True

@responses.activate
def test_load_model_failure(client):
    url = f"{client.management_url}/models/load"
    responses.add(
        responses.POST,
        url,
        json={"error": "Out of memory"},
        status=500
    )

    success = client.load_model("test-model", 32768, 1.0)
    assert success is False

@responses.activate
def test_unload_model(client):
    url = f"{client.base_url}/models/test-model"
    responses.add(
        responses.DELETE,
        url,
        status=200
    )

    success = client.unload_model("test-model")
    assert success is True

@patch("lm_studio_client.OpenAI")
def test_generate_with_metrics(mock_openai_class):
    # Setup mock OpenAI client
    mock_openai_instance = MagicMock()
    mock_openai_class.return_value = mock_openai_instance

    # Mock the stream response
    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices = [MagicMock(delta=MagicMock(content="Hello "))]

    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices = [MagicMock(delta=MagicMock(content="World"))]
    mock_chunk_2.usage = MagicMock(completion_tokens=2)

    mock_openai_instance.chat.completions.create.return_value = [mock_chunk_1, mock_chunk_2]

    client = LMStudioHardwareClient()
    result = client.generate_with_metrics("Say Hello World", "test-model")

    assert result["success"] is True
    assert result["output"] == "Hello World"
    assert result["tps"] > 0
    assert result["ttft"] >= 0
