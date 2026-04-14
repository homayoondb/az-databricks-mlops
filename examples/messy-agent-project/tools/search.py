# tool for searching stuff
# not wired up yet but keeping it here

def search_docs(query):
    """search internal docs - placeholder"""
    # TODO: hook this up to vector search
    return [
        {"title": "Getting Started Guide", "score": 0.95},
        {"title": "API Reference", "score": 0.82},
    ]


def search_web(query):
    # this one definitely doesnt work yet
    raise NotImplementedError("web search not implemented")
