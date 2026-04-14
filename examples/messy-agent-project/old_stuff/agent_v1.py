# DONT USE THIS - old version
# had a bug where it would loop forever on long questions
# keeping for reference only

def old_agent(question):
    if len(question) > 100:
        return old_agent(question[:100])  # the bug lol
    return f"I think the answer is: 42"
