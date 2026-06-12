import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from lm_studio_client import get_wsl_host_ip, is_wsl

load_dotenv()


def _default_base_url() -> str:
    host = get_wsl_host_ip() if is_wsl() else "127.0.0.1"
    return f"http://{host}:1234/v1"


class LLMEvaluator:
    def __init__(self, base_url=None, api_key=None, model=None):
        self.base_url = base_url or os.getenv("JUDGE_BASE_URL", _default_base_url())
        self.api_key = api_key or os.getenv("JUDGE_API_KEY", "lm-studio")
        self.model = model or os.getenv("JUDGE_MODEL", "local-model")

        # Use the standard OpenAI client. Since the URL and Key are
        # configurable, this can be pointed to any OpenAI-compatible provider
        # (LM Studio, LiteLLM, OpenAI, etc.)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def evaluate(self, prompt: str, rubric: str, model_output: str) -> dict:
        """
        Uses an LLM-as-a-judge to evaluate model_output against a rubric.

        Returns a dict with 'score' and 'reasoning'. 'score' is an int from
        1-10, or None when the judge's response could not be parsed or the
        request failed — callers must exclude None scores from aggregates.
        """
        system_prompt = (
            "You are an impartial expert judge evaluating an AI model's response. "
            "You will be given the original prompt, the required grading rubric, "
            "and the model's actual output. "
            "Evaluate the response carefully according to the rubric. "
            "Provide a brief reasoning for your evaluation, and then assign a "
            "final score from 1 to 10. "
            "Your final line MUST be exactly: 'SCORE: <number>' where <number> "
            "is an integer from 1 to 10."
        )

        user_content = (
            f"### ORIGINAL PROMPT:\n{prompt}\n\n"
            f"### RUBRIC:\n{rubric}\n\n"
            "### MODEL OUTPUT (delimited by <<< and >>>):\n"
            f"<<<\n{model_output}\n>>>\n\n"
            "Please evaluate the response. If the delimited output is empty "
            "or unrelated to the prompt, assign SCORE: 1."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,  # Low temperature for more consistent grading
            )

            judge_response = response.choices[0].message.content

            score_match = re.search(r"SCORE:\s*(\d+)", judge_response, re.IGNORECASE)

            if score_match:
                score = int(score_match.group(1))
                score = max(1, min(10, score))
            else:
                print(
                    "Warning: Judge did not format score correctly. "
                    f"Response: {judge_response}"
                )
                score = None

            return {
                "score": score,
                "reasoning": judge_response,
            }

        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                "score": None,
                "reasoning": f"Error: {e}",
            }


if __name__ == "__main__":
    evaluator = LLMEvaluator()
    sample_prompt = "Write a 3 sentence poem about a robot learning to feel."
    sample_rubric = (
        "The response must be exactly 3 sentences long. It must be a poem. "
        "It must be about a robot learning to feel emotions."
    )
    sample_output = (
        "I am a robot of steel and wire. A spark ignites within my core. "
        "I now know what it means to desire."
    )

    print("Testing Evaluator...")
    result = evaluator.evaluate(sample_prompt, sample_rubric, sample_output)
    print(f"Score: {result['score']}/10")
    print(f"Reasoning:\n{result['reasoning']}")
