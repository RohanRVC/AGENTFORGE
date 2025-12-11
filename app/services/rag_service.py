from typing import Dict
from qdrant_client import QdrantClient
from app.services.embedding_service import embed_text
from app.utils.latency import measure_latency
from app.utils.rouge_utils import compute_rouge_l

client = QdrantClient(host="qdrant", port=6333)
COLLECTION = "agentforge_embeddings"


@measure_latency("RAG Search")
def rag_search(query_text: str, top_k: int = 5) -> Dict:

    vector = embed_text(query_text)

   
    hits = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )

    results = []
    rouge_scores = []

    for h in hits.points:

        payload = h.payload or {}
        text = payload.get("text", "")

        rouge = compute_rouge_l(query_text, text)
        rouge_scores.append(rouge)

        results.append({
            "score": float(h.score),
            "rougeL": rouge,
            "doc_id": payload.get("doc_id"),
            "chunk_id": h.id,
            "text": text,
            "metadata": payload
        })

    sim_scores = [r["score"] for r in results]

    sim_stats = {
        "max_score": max(sim_scores) if sim_scores else 0,
        "min_score": min(sim_scores) if sim_scores else 0,
        "avg_score": sum(sim_scores) / len(sim_scores) if sim_scores else 0
    }

    hit_rate = (
        sum(1 for s in sim_scores if s >= 0.5) / len(sim_scores)
        if sim_scores else 0
    )

    rouge_stats = {
        "max_rouge": max(rouge_scores) if rouge_scores else 0,
        "min_rouge": min(rouge_scores) if rouge_scores else 0,
        "avg_rouge": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    }

    return {
        "results": results,
        "similarity_stats": sim_stats,
        "hit_rate": hit_rate,
        "rouge_stats": rouge_stats,
    }
