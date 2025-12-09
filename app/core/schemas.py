from pydantic import BaseModel
from typing import List, Optional

class IngestResponse(BaseModel):
    document_id: str
    status: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4
    use_agent: bool = False

class QueryResponse(BaseModel):
    answer: str
    context_used: List[str]
    steps: Optional[List[str]] = None


class AgentRequest(BaseModel):
    task: str

class AgentResponse(BaseModel):
    final_answer: str
    steps: List[str]
    metrics: dict
