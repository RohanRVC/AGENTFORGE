from typing import List, Dict


def build_rag_context(results: List[Dict], max_chars: int = 4000) -> str:
    """
    Build a clean context string from Qdrant search results.

    results = [
      {
        "doc_id": "...",
        "chunk_id": "...",
        "score": 0.89,
        "payload": {
            "type": "text" | "image" | "audio" | "video",
            "text": "...",
            "sequence": 0,
            ...
        }
      }
    ]
    """

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    seen = set()
    unique = []
    for r in sorted_results:
        cid = r["chunk_id"]
        if cid not in seen:
            unique.append(r)
            seen.add(cid)

    unique = sorted(unique, key=lambda x: x["payload"].get("sequence", 0))

    context_parts = []

    for r in unique:
        p = r["payload"]
        chunk_type = p.get("type", "text")
        text = p.get("text", "").strip()

        if not text:
            continue

        label = chunk_type.upper()

        context_parts.append(f"[{label}] {text}")

    context = "\n\n".join(context_parts)

    if len(context) > max_chars:
        context = context[:max_chars] + "\n...[trimmed]"

    return context
