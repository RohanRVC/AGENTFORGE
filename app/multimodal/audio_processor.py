import uuid
from typing import Dict

from sqlalchemy.orm import Session

from app.services.whisper_service import transcribe_audio
from app.utils.chunker import chunk_text
from app.services.embedding_service import embed_text
from app.services.qdrant_service import insert_embedding
from app.db import crud


def process_audio(doc_id: str, audio_path: str, db: Session) -> Dict:
    """
    Audio ingestion pipeline using faster-whisper.
    """

    transcript = transcribe_audio(audio_path)

    if not transcript:
        return {
            "doc_id": doc_id,
            "transcript": "",
            "chunk_count": 0
        }

    crud.update_document_transcript(db, doc_id, transcript)

    chunks = chunk_text(transcript, max_words=250, overlap_words=40)

    for idx, chunk_text_str in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        vector = embed_text(chunk_text_str)

        insert_embedding(
            doc_id=doc_id,
            chunk_id=chunk_id,
            vector=vector,
            metadata={
                "type": "audio",
                "source": "transcript",
                "chunk_index": idx,
                "text": chunk_text_str
            },
        )

        crud.create_chunk(
            db=db,
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=chunk_text_str,
            index=idx,
            vector_id=chunk_id,
        )

    return {
        "doc_id": doc_id,
        "transcript": transcript,
        "chunk_count": len(chunks)
    }
