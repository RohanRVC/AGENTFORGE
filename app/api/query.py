from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.rag_service import rag_search
from app.services.llm_service import run_llama_rag   


router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    context: list
    results: list
    metrics: dict


@router.post("/", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    rag = rag_search(request.question, top_k=request.top_k)

    if not rag["results"]:
        raise HTTPException(status_code=404, detail="No matching chunks found")

    context_text = "\n".join(
        [r["text"] for r in rag["results"] if r["text"]]
    )

    prompt = f"""
You are a RAG assistant.
User question: {request.question}

Context:
{context_text}

Answer clearly using only the context.
"""

    answer, cost_info = run_llama_rag(prompt)   # FIXED ✔

    return QueryResponse(
        answer=answer,
        context=context_text.split("\n"),
        results=rag["results"],
        metrics={
            "similarity_stats": rag["similarity_stats"],
            "hit_rate": rag["hit_rate"],
            "cost_info": cost_info,
            "rouge_stats": rag["rouge_stats"],
        }
    )
