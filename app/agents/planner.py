
from typing import List, Dict


def plan_steps(user_query: str) -> List[Dict]:
    """
    Rule-based planner for the agent.
    Ensures every step matches the controller format.
    """

    steps = []

    q = user_query.lower()

 
    rag_keywords = [
        "document", "docs", "according to", "context", "from file",
        "from ingestion", "knowledge base", "database", "rag", "what", "explain"
    ]

    if any(k in q for k in rag_keywords):
        steps.append({
            "tool": "rag",
            "input": user_query
        })

    
    calc_keywords = [
        "calculate", "sum", "minus", "percentage", "percent", "divide", "multiply"
    ]

    has_numbers = any(ch.isdigit() for ch in q)

    if has_numbers or any(k in q for k in calc_keywords):
        steps.append({
            "tool": "calculator",
            "input": user_query
        })

    
    web_keywords = [
        "website", "web", "internet", "url", "link", "fetch", "scrape", "visit"
    ]

    if any(k in q for k in web_keywords):
        steps.append({
            "tool": "web_scraper",
            "input": user_query
        })

   
    if len(steps) == 0:
        steps.append({
            "tool": "rag",
            "input": user_query
        })

    return steps
