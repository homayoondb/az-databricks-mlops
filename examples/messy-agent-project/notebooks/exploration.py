# exploring different prompt templates
# keeping this around for reference

TEMPLATES = {
    "default": "Answer the following question: {question}",
    "detailed": "You are a helpful assistant. Please provide a detailed answer to: {question}",
    "concise": "Answer in one sentence: {question}",
}

# tried these prompts, "detailed" works best but is slow
# "concise" sometimes cuts off important info
# sticking with "default" for now
