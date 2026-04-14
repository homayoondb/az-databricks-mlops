# quick test notebook - run this to check the agent works
# not a real test suite, just manual checks

import sys
sys.path.insert(0, "..")

from agent import agent

questions = [
    "What can you help me with?",
    "What is the capital of France?",
    "Tell me about Python",
    "What is Databricks?",
    "How do I make a sandwich?",
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {agent(q)}")
    print()
