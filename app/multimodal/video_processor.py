import os
import uuid
from typing import Dict, List

from sqlalchemy.orm import Session

from app.utils.video_utils import extract_keyframes_ffmpeg, extract_audio_ffmpeg
from app.services.llava_service import run_llava_caption
from app.services.embedding_service import embed_text
from app.services.qdrant_service import insert_embedding
from app.services.whisper_service import transcribe_audio
from app.utils.chunker import chunk_text
from app.db import crud


def process_video(doc_id: str, video_path: str, db: Session) -> Dict:
    """
    Full video ingestion pipeline.

    1. Extract keyframes with FFmpeg.
    2. Caption each frame with LLaVA (pip ollama).
    3. Embed each caption and push to Qdrant + Postgres.
    4. Extract audio track from the video.
    5. Transcribe audio with Whisper (pip ollama).
    6. Chunk + embed transcript, push to Qdrant + Postgres.
    7. Store transcript on the Document row.

    Returns a small summary dict.
    """

    # ---------------------------
    # Part 1: Frames + captions
    # ---------------------------
    frames_dir = os.path.join("uploaded_files", "video_frames", doc_id)
    os.makedirs(frames_dir, exist_ok=True)

    frame_paths: List[str] = extract_keyframes_ffmpeg(
        video_path,
        frames_dir,
        seconds_interval=3,
    )

    frame_results = []

    for idx, frame_path in enumerate(frame_paths):
        caption = run_llava_caption(frame_path)
        if not caption:
            continue

        vector = embed_text(caption)

        chunk_id = str(uuid.uuid4())

        # Store in Qdrant
        insert_embedding(
            doc_id=doc_id,
            chunk_id=chunk_id,
            vector=vector,
            metadata={
                "type": "video_frame",
                "source": "frame_caption",
                "frame_index": idx,
                "file_path": frame_path,
            },
        )

        crud.create_chunk(
            db=db,
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=caption,
            index=idx,
            vector_id=chunk_id,
        )

        frame_results.append(
            {
                "frame_index": idx,
                "frame_path": frame_path,
                "caption": caption,
            }
        )

    audio_output_path = os.path.join(
        "uploaded_files",
        "video_audio",
        f"{doc_id}.wav",
    )
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)

    extract_audio_ffmpeg(video_path, audio_output_path)

    transcript = transcribe_audio(audio_output_path)

    chunk_count = 0

    if transcript:
        crud.update_document_transcript(db, doc_id, transcript)

        transcript_chunks = chunk_text(
            transcript,
            max_words=250,
            overlap_words=40,
        )

        for idx, chunk_text_str in enumerate(transcript_chunks):
            chunk_id = str(uuid.uuid4())
            vector = embed_text(chunk_text_str)

            insert_embedding(
                doc_id=doc_id,
                chunk_id=chunk_id,
                vector=vector,
                metadata={
                    "type": "video_audio",
                    "source": "transcript",
                    "chunk_index": idx,
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

        chunk_count = len(transcript_chunks)

    return {
        "doc_id": doc_id,
        "frame_count": len(frame_paths),
        "audio_transcript_present": bool(transcript),
        "transcript_chunk_count": chunk_count,
    }


def process_video_for_query(video_path: str) -> str:
    """
    Simple video processor for the /multimodal endpoint.

    - Extracts a few frames and captions them with LLaVA.
    - Extracts audio and transcribes it with Whisper.
    - Returns a combined text summary (no embeddings, no DB).
    """

    frames_dir = "tmp_query_frames"
    os.makedirs(frames_dir, exist_ok=True)

    frame_paths = extract_keyframes_ffmpeg(
        video_path,
        frames_dir,
        seconds_interval=3,
    )

    frame_captions = []
    for idx, fp in enumerate(frame_paths):
        caption = run_llava_caption(fp)
        if caption:
            frame_captions.append(f"Frame {idx}: {caption}")

    tmp_audio_path = "tmp_query_audio.wav"
    extract_audio_ffmpeg(video_path, tmp_audio_path)
    transcript = transcribe_audio(tmp_audio_path)

    frames_text = (
        "\n".join(frame_captions)
        if frame_captions
        else "No frame captions available."
    )
    transcript_text = transcript if transcript else "No transcript available."

    combined = f"""
Frames:
{frames_text}

Audio:
{transcript_text}
""".strip()

    return combined
