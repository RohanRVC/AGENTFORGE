
import os
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from typing import List, Dict, Optional

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

COLLECTION_NAME = "agentforge_embeddings"

client = None


def init_qdrant():
    """
    Safe Qdrant initialization with retries.
    Prevents API from crashing if Qdrant starts slowly.
    """
    global client

    for attempt in range(1, 11):  # 10 retries
        try:
            print(f"[QDRANT] Trying to connect ({attempt}/10) to {QDRANT_HOST}:{QDRANT_PORT}")

            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

            # Simple check
            collections = client.get_collections().collections

            print("[QDRANT] Connection successful.")

            # Create collection if missing
            existing = [c.name for c in collections]
            if COLLECTION_NAME not in existing:
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qmodels.VectorParams(
                        size=768,
                        distance=qmodels.Distance.COSINE
                    ),
                )
                print(f"✅ Created Qdrant collection: {COLLECTION_NAME}")
            else:
                print(f"ℹ️ Collection already exists: {COLLECTION_NAME}")

            return client

        except Exception as e:
            print(f"[QDRANT] Not ready yet → {e}")
            time.sleep(2)

    print("[QDRANT] FAILED after retries. Continuing without Qdrant.")
    client = None
    return None


def get_client():
    return client


def insert_embedding(doc_id: str, chunk_id: str, vector: list, metadata: dict):
    if client is None:
        raise RuntimeError("Qdrant client not initialized")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            qmodels.PointStruct(
                id=chunk_id,
                vector=vector,
                payload={"doc_id": doc_id, **metadata},
            )
        ],
    )


def search_similar(query_vector: List[float], top_k: int = 5, filter_doc_id: Optional[str] = None):
    if client is None:
        raise RuntimeError("Qdrant client not initialized")

    query_filter = None
    if filter_doc_id:
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_id",
                    match=qmodels.MatchValue(value=filter_doc_id),
                )
            ]
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            "doc_id": point.payload.get("doc_id"),
            "chunk_id": point.id,
            "score": point.score,
            "payload": point.payload,
        }
        for point in results
    ]
