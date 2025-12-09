from fastapi import APIRouter, File, UploadFile, Form
from app.core.schemas import QueryRequest, QueryResponse
from app.services.llama_service import run_llama
from app.services.llava_service import run_llava_caption
from app.services.whisper_service import transcribe_audio
import os

router = APIRouter()

@router.post("/")
async def query(
    query: str = Form(...),
    file: UploadFile | None = File(None)
):
    if file:
        ext = file.filename.split(".")[-1].lower()
        filepath = f"uploaded_files/{file.filename}"

        with open(filepath, "wb") as f:
            f.write(await file.read())

        if ext in ["jpg", "jpeg", "png"]:
            answer = run_llava(query, filepath)

        elif ext in ["mp3", "wav", "m4a"]:
            text = transcribe_audio(filepath)
            answer, cost_info= run_llama(query + "\n\nAudio content:\n" + text)

        elif ext in ["mp4", "mov"]:
            answer = "Video support coming soon."

        else:
            answer = run_llama(query)

    else:
        answer, cost_info = run_llama(query)

    return QueryResponse(
        answer=answer,
        context_used=[],
        steps=[]
    )
