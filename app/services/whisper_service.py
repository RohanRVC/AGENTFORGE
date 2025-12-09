import requests
from app.utils.latency import measure_latency
from app.eval.cost_metrics import estimate_cost


OLLAMA_URL = "http://localhost:11434/api/generate"
WHISPER_MODEL = "whisper"


@measure_latency("Whisper Transcription")
def transcribe_audio(audio_path: str) -> str:
    """
    Fully correct Whisper transcription using Ollama's required format.
    """

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        files = {
            "input": ("audio.wav", audio_bytes, "application/octet-stream")
        }

        data = {
            "model": WHISPER_MODEL,
            "prompt": "transcribe"  
        }

        response = requests.post(
            OLLAMA_URL,
            data=data,
            files=files,
            timeout=300
        )

        raw = response.text.strip()

        transcript = raw.replace("\n", " ").strip()

        if "error" in transcript.lower():
            print("[Whisper ERROR] Whisper returned system error:")
            print(transcript)
            return ""

        output_tokens = len(transcript.split())
        _ = estimate_cost("whisper", 0, output_tokens)

        return transcript

    except Exception as e:
        print(f"[Whisper Exception] {e}")
        return ""
