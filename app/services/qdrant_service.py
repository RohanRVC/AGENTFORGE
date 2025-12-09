from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from typing import List, Dict, Optional


qdrant = QdrantClient(
    host="localhost",
    port=6333,
)

COLLECTION_NAME = "agentforge_embeddings"


def init_qdrant():
    """
    Create the Qdrant collection if it does not exist.
    Called from main.py on startup.
    """

    collections = qdrant.get_collections().collections
    existing_names = [c.name for c in collections]

    if COLLECTION_NAME not in existing_names:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=768,
                distance=qmodels.Distance.COSINE
            )
        )
        print(f"✅ Qdrant collection created: {COLLECTION_NAME}")
    else:
        print(f"ℹ️ Qdrant collection already exists: {COLLECTION_NAME}")



def insert_embedding(doc_id: str, chunk_id: str, vector: list, metadata: dict):
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            qmodels.PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    **metadata
                },
            )
        ],
    )


def search_similar(
    query_vector: List[float],
    top_k: int = 5,
    filter_doc_id: Optional[str] = None,
) -> List[Dict]:

    query_filter = None
    if filter_doc_id:
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_id",
                    match=qmodels.MatchValue(value=filter_doc_id)
                )
            ]
        )

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )

    output = []
    for point in results:
        output.append(
            {
                "doc_id": point.payload.get("doc_id"),
                "chunk_id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
        )

    return output
