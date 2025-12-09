import ollama
from app.utils.latency import measure_latency
from app.eval.cost_metrics import estimate_cost


LLAMA_MODEL_NAME = "llama3.1"  


@measure_latency("LLaMA RAG Inference")
def run_llama_rag(prompt: str):
    """
    RAG-specific LLaMA call using pip Ollama SDK.
    Returns:
        - answer_text
        - simulated cost metrics
    """
    try:

        result = ollama.generate(
            model=LLAMA_MODEL_NAME,
            prompt=prompt,
            stream=False
        )

        answer = result.get("response", "").strip()

       
        input_tokens = len(prompt.split())
        output_tokens = len(answer.split())

        cost_info = estimate_cost(
            model=LLAMA_MODEL_NAME,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        return answer, cost_info

    except Exception as e:
        print(f"[LLaMA RAG Error] {e}")
        return (
            "Sorry, the model could not generate an answer right now.",
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0
            }
        )
