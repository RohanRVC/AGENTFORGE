from app.services.llava_service import run_llava_caption
from app.services.embedding_service import embed_text


def process_image(image_path: str) -> dict:
    """
    Full image ingestion pipeline.

    1. Generate a caption for the image using LLaVA.
    2. Embed the caption text.
    3. Return both caption and embedding so the caller
       can push to Qdrant + Postgres.
    """

    caption = run_llava_caption(image_path)

    if not caption:
        caption = "No description available for this image."

    embedding = embed_text(caption)

    return {
        "caption": caption,
        "embedding": embedding,
    }


def process_image_for_query(image_path: str) -> str:
    """
    Lightweight helper for the /multimodal query endpoint.

    - Generate a caption for the given image file.
    - Return plain caption text (no embeddings, no DB writes).
    """

    caption = run_llava_caption(image_path)

    if not caption:
        caption = "No description available from the image."

    return caption.strip()
