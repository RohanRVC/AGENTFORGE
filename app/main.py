from fastapi import FastAPI
from app.api.ingest_router import router as ingest_router
from app.api.query import router as query_router
from app.api.health import router as health_router
from app.api.agent import router as agent_router
from app.api.multimodal import router as multimodal_router
from app.services.qdrant_service import init_qdrant
from qdrant_client import QdrantClient

q = QdrantClient(host="localhost", port=6333)
q.delete_collection("agentforge_embeddings")

app = FastAPI(
    title="AgentForge",
    description="Agentic RAG + Multimodal AI Backend",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    init_qdrant()   


app.include_router(health_router, prefix="/health")
app.include_router(ingest_router, prefix="/ingest")
app.include_router(query_router, prefix="/query")
app.include_router(agent_router, prefix="/agent")
app.include_router(multimodal_router, prefix="/multimodal")
