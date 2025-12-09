import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import crud

from app.multimodal.image_processor import process_image
from app.multimodal.audio_processor import process_audio
from app.multimodal.video_processor import process_video

from app.utils.chunker import chunk_text
from app.services.embedding_service import embed_text
from app.services.qdrant_service import insert_embedding

from pypdf import PdfReader  

router = APIRouter()


def save_uploaded_file(upload_file: UploadFile, upload_dir: str) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, upload_file.filename)

    upload_file.file.seek(0)
    with open(file_path, "wb") as f:
        f.write(upload_file.file.read())

    return file_path



@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):

    db: Session = SessionLocal()
    doc_id = str(uuid.uuid4())

    upload_dir = "uploaded_files"
    saved_path = save_uploaded_file(file, upload_dir)

    filename = file.filename
    ext = filename.split(".")[-1].lower()

   
    TEXT_EXT = {"txt"}
    PDF_EXT = {"pdf"}
    IMAGE_EXT = {"jpg", "jpeg", "png", "webp", "bmp", "gif", "heic"}
    AUDIO_EXT = {"mp3", "wav", "m4a", "aac", "flac", "ogg"}
    VIDEO_EXT = {"mp4", "mov", "avi", "mkv", "webm"}

    
    extracted_text = None

    if ext in PDF_EXT:
        try:
            reader = PdfReader(saved_path)
            extracted_text = ""

            for page in reader.pages:
                page_text = page.extract_text() or ""
                extracted_text += page_text + "\n"

            if not extracted_text.strip():
                raise HTTPException(400, "PDF contains no extractable text.")

        except Exception as e:
            raise HTTPException(500, f"PDF extraction failed: {str(e)}")

        doc_type = "text"

 
    elif ext in TEXT_EXT:
        doc_type = "text"

    elif ext in IMAGE_EXT:
        doc_type = "image"

    elif ext in AUDIO_EXT:
        doc_type = "audio"

    elif ext in VIDEO_EXT:
        doc_type = "video"

    else:
        raise HTTPException(400, f"Unsupported file type: .{ext}")


    crud.create_document(db, doc_id, doc_type, saved_path)



    if doc_type == "text":

        if extracted_text is not None:
            text = extracted_text.strip()   
        else:
            with open(saved_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

        if not text:
            raise HTTPException(400, "Text content is empty.")

        chunks = chunk_text(text, max_words=250, overlap_words=40)

        for idx, text_chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            vector = embed_text(text_chunk)

            insert_embedding(
                doc_id=doc_id,
                chunk_id=chunk_id,
                vector=vector,
                metadata={
                    "type": "text",
                    "chunk_index": idx,
                    "text": text_chunk,
                    "source": "pdf" if extracted_text is not None else "txt"
                }
            )

            crud.create_chunk(
                db=db,
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text_chunk,
                index=idx,
                vector_id=chunk_id
            )

        return {
            "doc_id": doc_id,
            "type": "text",
            "chunks": len(chunks),
            "source": "pdf → text (in memory)" if extracted_text else "text"
        }



    
    if doc_type == "image":
        result = process_image(saved_path)
        caption = result["caption"]
        vector = result["embedding"]

        chunk_id = str(uuid.uuid4())

        insert_embedding(
            doc_id=doc_id,
            chunk_id=chunk_id,
            vector=vector,
            metadata={"type": "image", "source": "caption", "text": caption},
        )

        crud.update_document_caption(db, doc_id, caption)
        crud.create_chunk(db, chunk_id, doc_id, caption, 0, chunk_id)

        return {"doc_id": doc_id, "type": "image", "caption": caption}



    # ============================================================
    # AUDIO INGESTION
    # ============================================================
    if doc_type == "audio":
        result = process_audio(doc_id, saved_path, db)
        return {"doc_id": doc_id, "type": "audio", **result}



    # ============================================================
    # VIDEO INGESTION
    # ============================================================
    if doc_type == "video":
        result = process_video(doc_id, saved_path, db)
        return {"doc_id": doc_id, "type": "video", **result}
