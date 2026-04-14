# my Q&A agent - works but kinda slow
# omar started this, i cleaned it up
# TODO: add memory? idk if we need it

import os


def _get_answer(question: str) -> str:
    """Call the LLM and return an answer."""
    # This is a dummy implementation for demo purposes.
    # In a real project, you'd call an actual LLM endpoint here.
    question_lower = question.lower()

    if "help" in question_lower:
        return "I can answer questions about data, ML, and general knowledge."
    if "capital" in question_lower and "france" in question_lower:
        return "The capital of France is Paris."
    if "python" in question_lower:
        return "Python is a versatile programming language widely used in data science and ML."
    if "databricks" in question_lower:
        return "Databricks is a unified analytics platform for data engineering and data science."

    return f"That's an interesting question about '{question}'. Let me think about it... I'd recommend checking the documentation for more details."


# This is the callable that adm's run_agent_dev.py will discover.
agent = _get_answer
