
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
from app.agents.controller import run_agent_controller
from app.agents.langgraph_agent import run_langgraph_agent   
from app.core.schemas import AgentResponse

router = APIRouter()


class AgentRequest(BaseModel):
    task: str
    engine: str = "controller"   


@router.post("/", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    Run agentic reasoning pipeline.
    """

    try:
        engine = request.engine.lower()

       
        if engine == "controller":
            result = run_agent_controller(request.task)

            return AgentResponse(
                final_answer=result.get("final_answer", ""),
                steps=result.get("steps", []),
                metrics=result.get("metrics", {})
            )

   
        elif engine == "langgraph":
            raw = run_langgraph_agent(request.task)

            if not isinstance(raw, dict):
                raw = {"final_answer": str(raw)}

            final_answer = (
                raw.get("final_answer")
                or raw.get("answer")
                or raw.get("output")
                or raw.get("result")
                or json.dumps(raw)   
            )

            steps = (
                raw.get("steps")
                or raw.get("trace")
                or []
            )

            metrics = (
                raw.get("metrics")
                or raw.get("stats")
                or {}
            )

            return AgentResponse(
                final_answer=final_answer,
                steps=steps,
                metrics=metrics
            )


        else:
            raise HTTPException(status_code=400, detail="Invalid engine type.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
