from typing import List

def clean_text(text: str) -> str:
    """
    Clean raw text by normalizing spaces and newlines.
    This helps make chunking more stable.
    """
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
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
        chunks.append(" ".join(words))
        return chunks

    start = 0
    end = max_words

    while start < len(words):
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap_words
        end = start + max_words

    return chunks
