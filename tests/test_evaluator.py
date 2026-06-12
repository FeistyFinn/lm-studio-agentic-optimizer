from unittest.mock import MagicMock, patch

import pytest

from evaluator import LLMEvaluator


@pytest.fixture
def evaluator():
    with patch("evaluator.OpenAI"):
        yield LLMEvaluator(base_url="http://judge:1234/v1", api_key="k", model="judge")


def respond_with(evaluator, content):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    evaluator.client.chat.completions.create.return_value = response


def test_parses_score(evaluator):
    respond_with(evaluator, "Solid response, minor issues.\nSCORE: 8")

    result = evaluator.evaluate("prompt", "rubric", "output")

    assert result["score"] == 8
    assert "Solid response" in result["reasoning"]


def test_clamps_out_of_range_score(evaluator):
    respond_with(evaluator, "Outstanding.\nSCORE: 15")

    assert evaluator.evaluate("p", "r", "o")["score"] == 10


def test_unparseable_response_returns_none(evaluator):
    respond_with(evaluator, "I would rate this rather highly, perhaps an eight.")

    assert evaluator.evaluate("p", "r", "o")["score"] is None


def test_api_error_returns_none(evaluator):
    evaluator.client.chat.completions.create.side_effect = RuntimeError("down")

    result = evaluator.evaluate("p", "r", "o")

    assert result["score"] is None
    assert "down" in result["reasoning"]
