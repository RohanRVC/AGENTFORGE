
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup

from app.services.rag_service import rag_search
from app.services.llama_service import run_llama   # <-- UPDATED


def rag_tool(question: str) -> Dict[str, Any]:

    rag = rag_search(question)

    context_text = "\n".join([r["text"] for r in rag["results"] if r["text"]])

    prompt = f"""
Use the following context to answer the question.

Question:
{question}

Context:
{context_text}

Give a clear and helpful answer.
"""

    answer, cost_info = run_llama(prompt)   # <-- UPDATED

    return {
        "response": {
            "answer": answer,
            "context": rag["results"],
            "metrics": rag
        }
    }



def calculator_tool(expression: str) -> Dict[str, Any]:
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return {"response": str(result)}
    except Exception as e:
        return {"response": f"CALC_ERROR: {e}"}



def web_scraper_tool(url: str) -> Dict[str, Any]:
    try:
        page = requests.get(url, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")
        text = soup.get_text(separator="\n")
        cleaned = text.strip()[:1500]
        return {"response": cleaned}
    except Exception as e:
        return {"response": f"WEB_ERROR: {e}"}
