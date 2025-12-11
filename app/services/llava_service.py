import ollama
from app.utils.latency import measure_latency


LLAVA_MODEL = "llava:7b"   


@measure_latency("LLaVA Captioning")
def run_llava_caption(image_path: str) -> str:
    """
    Image captioning using the LLaVA multimodal model via pip Ollama SDK.
    Supports all normal image formats: JPG, PNG, JPEG, WEBP.
    """

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        result = ollama.generate(
            model=LLAVA_MODEL,
            prompt="Describe this image in one clear and accurate sentence.",
            images=[image_bytes],
            stream=False
        )

        caption = result.get("response", "").strip()
        return caption if caption else ""

    except Exception as e:
        print(f"[LLaVA Error] {e}")
        return ""


@measure_latency("LLaVA VQA")
def run_llava_vqa(image_path: str, question: str) -> str:
    """
    Visual Question Answering:
    Ask a question about an image.
    """

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        result = ollama.generate(
            model=LLAVA_MODEL,
            prompt=f"Q: {question}\nA:",
            images=[image_bytes],
            stream=False
        )

        return result.get("response", "").strip()

    except Exception as e:
        print(f"[LLaVA VQA Error] {e}")
        return ""
