import ollama
from app.utils.latency import measure_latency
from app.eval.cost_metrics import estimate_cost


LLAMA_MODEL_NAME = "llama3.1"  


@measure_latency("LLaMA Inference")
def run_llama(prompt: str):
    """
    Uses the pip Ollama SDK (stable on Windows).
    This is the main LLM used across:
    - RAG (run_llama_rag uses this internally)
    - Agents
    - Multimodal final answer
    """

    try:
     
        result = ollama.generate(
            model=LLAMA_MODEL_NAME,
            prompt=prompt,
            stream=False
        )

        answer = result.get("response", "").strip()

        clean = " ".join(answer.split())  

    
        input_tokens = len(prompt.split())
        output_tokens = len(clean.split())

        cost_info = estimate_cost(
            model=LLAMA_MODEL_NAME,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        return clean, cost_info


    except Exception as e:
        print(f"[LLaMA SDK Error] {e}")

        return (
            "Sorry, the model could not generate an answer right now.",
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0
            }
        )
