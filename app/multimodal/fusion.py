def fuse_modalities(
    query: str,
    caption: str | None,
    transcript: str | None,
    video_text: str | None
) -> str:
    """
    Combine all available modes to enrich the RAG search.
    """

    fused_parts = [query]

    if caption:
        fused_parts.append(f"Image: {caption}")

    if transcript:
        fused_parts.append(f"Audio: {transcript}")

    if video_text:
        fused_parts.append(f"Video: {video_text}")

    return " ".join(fused_parts)
