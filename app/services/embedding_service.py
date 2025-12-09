from sentence_transformers import SentenceTransformer
from typing import List

_embedding_model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1",
    trust_remote_code=True
)

def embed_text(text: str) -> List[float]:
    """
    Create an embedding vector using Nomic embed text model.
    """
    if not text or not text.strip():
        return [0.0] * 768  # fallback vector

    vector = _embedding_model.encode(text, convert_to_numpy=True).tolist()
    return vector
