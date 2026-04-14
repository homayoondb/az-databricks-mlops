"""Agent evaluation scorers for mlflow.genai.evaluate().

This module defines scorer functions that judge agent outputs.
Customize the scorers below for your use case.

Scorer types:
  - Rule-based: deterministic checks (fast, free, reliable)
  - LLM-as-judge: uses a model to assess quality (flexible, handles nuance)

Each scorer receives (inputs, outputs, expectations) and returns a score dict.
"""

from mlflow.genai.scorers import scorer


@scorer
def response_not_empty(inputs, outputs, expectations=None):
    """Check that the agent produced a non-empty response."""
    if not outputs or not str(outputs).strip():
        return {"score": 0, "justification": "Response was empty."}
    return {"score": 1, "justification": "Response is non-empty."}


@scorer
def response_length_check(inputs, outputs, expectations=None):
    """Flag responses that are suspiciously short or excessively long."""
    text = str(outputs).strip()
    length = len(text)
    if length < 10:
        return {"score": 0, "justification": f"Response too short ({length} chars)."}
    if length > 10000:
        return {"score": 0, "justification": f"Response too long ({length} chars)."}
    return {"score": 1, "justification": f"Response length OK ({length} chars)."}


# --- LLM-as-judge scorers (uncomment and configure when ready) ---
#
# from mlflow.genai.scorers import llm_judge
#
# relevance = llm_judge(
#     name="relevance",
#     prompt="Rate the relevance of the response to the input on a scale of 1-5.",
#     model="endpoints:/databricks-gpt-5-4",
# )
#
# correctness = llm_judge(
#     name="correctness",
#     prompt=(
#         "Given the expected response and the actual response, "
#         "rate correctness on a scale of 1-5."
#     ),
#     model="endpoints:/databricks-gpt-5-4",
# )


def get_scorers():
    """Return the list of scorers to use in evaluation.

    Add or remove scorers here. Each scorer will be applied to every
    row in the evaluation dataset.
    """
    return [
        response_not_empty,
        response_length_check,
    ]
