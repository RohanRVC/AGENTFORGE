from fastapi import APIRouter, File, UploadFile, Form
from pydantic import BaseModel
import os

from app.services.llava_service import run_llava_caption
from app.services.whisper_service import transcribe_audio
from app.services.llama_service import run_llama
from app.services.rag_service import rag_search

from app.multimodal.image_processor import process_image_for_query
from app.multimodal.video_processor import process_video_for_query
from app.multimodal.fusion import fuse_modalities
from app.multimodal.final_prompt_builder import build_multimodal_prompt

from app.utils.latency import measure_latency


router = APIRouter()


class MultiModalResponse(BaseModel):
    final_answer: str
    image_caption: str | None
    audio_transcript: str | None
    video_summary: str | None
    rag_context: list
    rag_metrics: dict


@router.post("/")
@measure_latency("Multimodal Query")
async def multimodal_query(
    query: str = Form(...),
    file: UploadFile | None = File(None),
):
    caption = None
    transcript = None
    video_text = None

    if file:
        ext = file.filename.split(".")[-1].lower()
        filepath = f"uploaded_files/{file.filename}"

        os.makedirs("uploaded_files", exist_ok=True)

        with open(filepath, "wb") as f:
            f.write(await file.read())

        if ext in ["png", "jpg", "jpeg", "webp"]:
            caption = process_image_for_query(filepath)

        elif ext in ["mp3", "wav", "m4a", "aac", "flac", "ogg"]:
            transcript = transcribe_audio(filepath)

        elif ext in ["mp4", "mov", "avi", "mkv", "webm"]:
            video_text = process_video_for_query(filepath)

    
    fused_query = fuse_modalities(query, caption, transcript, video_text)

    rag = rag_search(fused_query)

    final_prompt = build_multimodal_prompt(
        query=query,
        caption=caption,
        transcript=transcript,
        video_text=video_text,
        rag_results=rag["results"]
    )

    final_answer, _ = run_llama(final_prompt)


    return MultiModalResponse(
        final_answer=final_answer,
        image_caption=caption,
        audio_transcript=transcript,
        video_summary=video_text,
        rag_context=rag["results"],
        rag_metrics={
            "similarity_stats": rag["similarity_stats"],
            "hit_rate": rag["hit_rate"],
            "rouge_stats": rag["rouge_stats"],
        },
    )
