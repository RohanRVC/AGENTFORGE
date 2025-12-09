def build_multimodal_prompt(query, caption=None, transcript=None, video_text=None, rag_results=None):
    context_text = ""

    if caption:
        context_text += f"\nImage Description:\n{caption}\n"

    if transcript:
        context_text += f"\nAudio Transcript:\n{transcript}\n"

    if video_text:
        context_text += f"\nVideo Summary:\n{video_text}\n"

    if rag_results:
        merged = "\n".join([r["text"] for r in rag_results if r.get("text")])
        context_text += f"\nRetrieved Document Context:\n{merged}\n"

    prompt = f"""
You are an expert multimodal reasoning assistant.

Use the image description and the document context to answer the user’s question clearly and directly. 
Do NOT mention RAG, retrieval, steps, or system reasoning.

Respond in a smooth, natural explanation suitable for humans.

User Question:
{query}

Relevant Context:
{context_text}

Final Answer:
"""

    return prompt.strip()
