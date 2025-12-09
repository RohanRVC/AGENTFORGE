# app/utils/chunker.py

from typing import List

def clean_text(text: str) -> str:
    """
    Clean raw text by normalizing spaces and newlines.
    This helps make chunking more stable.
    """
    # Replace newlines with spaces
    text = text.replace("\n", " ")
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    # Collapse multiple spaces into one
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def chunk_text(
    text: str,
    max_words: int = 250,
    overlap_words: int = 40
) -> List[str]:
    """
    Split a long text into overlapping chunks based on word count.

    - max_words: target size of each chunk.
    - overlap_words: number of words to repeat from the end of one chunk
      at the start of the next chunk.
    """
    text = clean_text(text)
    if not text:
        return []

    words = text.split(" ")
    chunks: List[str] = []

    if len(words) <= max_words:
        # Text is short, single chunk is enough
        chunks.append(" ".join(words))
        return chunks

    start = 0
    end = max_words

    while start < len(words):
        # Slice the words for this chunk
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)

        # Move the window forward with overlap
        if end >= len(words):
            break

        start = end - overlap_words
        end = start + max_words

    return chunks
