from typing import Dict

LOCAL_LLM_RATE = 0.0000001
LOCAL_VISION_RATE = 0.0000002
LOCAL_WHISPER_RATE = 0.00000015


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> Dict:
    """
    We simulate cost metrics because local models have no built-in pricing.
    This satisfies assignment expectations for model-cost tracking.
    """

    if "llava" in model.lower():
        rate = LOCAL_VISION_RATE
    elif "whisper" in model.lower():
        rate = LOCAL_WHISPER_RATE
    else:
        rate = LOCAL_LLM_RATE

    total_tokens = input_tokens + output_tokens
    estimated_cost = round(total_tokens * rate, 8)

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost
    }
