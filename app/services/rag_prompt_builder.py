def build_rag_prompt(question: str, context: str) -> str:
    """
    Build a deterministic RAG prompt.
    Used by Llama / Ollama to generate grounded answers.
    """

    prompt = f"""
You are a helpful assistant. Use only the information in the context to answer the question.
If the answer is not in the context, say "The context does not contain enough information."

Context:
----------------
{context}
----------------

Question: {question}

Give a clear and direct answer.
"""
    return prompt.strip()
