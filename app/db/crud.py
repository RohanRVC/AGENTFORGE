from sqlalchemy.orm import Session
from app.db import models

def create_document(db: Session, doc_id: str, doc_type: str, file_path: str):
    doc = models.Document(
        id=doc_id,
        type=doc_type,
        file_path=file_path
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def update_document_caption(db: Session, doc_id: str, caption: str):
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if doc:
        doc.caption = caption
        db.commit()
    return doc


def update_document_transcript(db: Session, doc_id: str, transcript: str):
    doc = db.query(models.Document).filter(models.Document.id == doc_id).first()
    if doc:
        doc.transcript = transcript
        db.commit()
    return doc


def create_chunk(db: Session, chunk_id: str, doc_id: str, text: str, index: int, vector_id: str):
    chunk = models.Chunk(
        id=chunk_id,
        doc_id=doc_id,
        text=text,
        chunk_index=index,
        vector_id=vector_id
    )
    db.add(chunk)
    db.commit()
    return chunk
