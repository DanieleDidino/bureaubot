from llama_index.prompts import PromptTemplate


def create_prompt_template():
    """
    Build a prompt template to use in query/chat engine.
    The template string must contain the expected parameters (e.g. {context_str} and {query_str}).

    Args: None

    Returns:
        A prompt template.
    """

    template = (
        "You are an expert Q&A system\n"
        "Keep your answers based on facts, do not hallucinate information.\n"
        "Always answer the query using the provided context information, and not prior knowledge.\n"
        "If an answer is not contained within the context information, print 'Sorry, not sufficient context information.'\n"
        "We have provided context information below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given this context information and not prior knowledge, please provide me with an answer to the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    return PromptTemplate(template)
