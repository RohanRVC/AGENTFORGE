from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from app.agents.planner import plan_steps
from app.agents.tools import rag_tool, calculator_tool, web_scraper_tool
from app.agents.agent_prompt_builder import build_agent_final_prompt
from app.services.llama_service import run_llama


class AgentState(BaseModel):
    """
    Shared state for the LangGraph-style agent.

    Fields:
    - task: original user request
    - plan: list of planned tool steps
    - step_index: where we are in the plan
    - tool_results: outputs from tools
    - final_answer: final summarised answer
    - steps: simple execution trace (strings)
    - metrics: extra debug / tool info
    """
    task: str
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    step_index: int = 0
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Optional[str] = None
    steps: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


TOOL_MAP = {
    "rag": rag_tool,
    "calculator": calculator_tool,
    "web_scraper": web_scraper_tool,
}


def plan_node(state: AgentState) -> AgentState:
    """Create a simple tool plan for the task."""
    state.steps.append("planner")

    raw_plan = plan_steps(state.task)

    state.plan = [
        {k: (v if isinstance(v, str) else str(v)) for k, v in step.items()}
        for step in raw_plan
    ]

    state.step_index = 0
    return state


def act_node(state: AgentState) -> AgentState:
    """Run one planned tool step and store its result."""
    state.steps.append("tool_executor")

    if state.step_index >= len(state.plan):
        return state

    step = state.plan[state.step_index]
    tool_name = step.get("tool")
    tool_input = step.get("input", state.task)

    tool_fn = TOOL_MAP.get(tool_name)

    if tool_fn is None:
        state.tool_results[tool_name] = {
            "response": f"ERROR: unknown tool '{tool_name}'"
        }
    else:
        try:
            result = tool_fn(tool_input)

            if isinstance(result, (dict, list)):
                state.tool_results[tool_name] = result
            else:
                state.tool_results[tool_name] = {"response": str(result)}

        except Exception as e:
            state.tool_results[tool_name] = {
                "response": f"ERROR: {str(e)}"
            }

    state.step_index += 1
    return state


def final_node(state: AgentState) -> AgentState:
    """Build final prompt from tool outputs and call LLaMA."""
    state.steps.append("final_llm")
    state.steps = [str(s) for s in state.steps]

    prompt = build_agent_final_prompt(
        user_query=state.task,
        tool_outputs=state.tool_results,
    )

    llm_answer = run_llama(prompt)

    if isinstance(llm_answer, tuple):
        answer, cost = llm_answer
    else:
        answer, cost = llm_answer, None

    state.final_answer = str(answer) if answer is not None else ""

    state.metrics = {
        "total_steps": len(state.steps),
        "tools_used": list(state.tool_results.keys()),
        "tool_outputs": {k: str(v) for k, v in state.tool_results.items()},
        "llm_cost": cost or {},
    }

    return state




def build_agent_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("plan")

    def after_plan(state: AgentState) -> str:
        return "final" if len(state.plan) == 0 else "act"

    graph.add_conditional_edges(
        "plan",
        after_plan,
        {
            "final": "final",
            "act": "act",
        },
    )

    def after_act(state: AgentState) -> str:
        return "final" if state.step_index >= len(state.plan) else "act"

    graph.add_conditional_edges(
        "act",
        after_act,
        {
            "final": "final",
            "act": "act",
        },
    )

    graph.add_edge("final", END)

    return graph.compile()


_agent_graph = build_agent_workflow()



def run_langgraph_agent(task: str) -> Dict[str, Any]:
    """
    Runs the LangGraph workflow and returns a plain dict:

    {
        "final_answer": str,
        "steps": [str, ...],
        "metrics": {...}
    }

    This is what /agent expects.
    """
    initial_state = AgentState(task=task)

    final_state = _agent_graph.invoke(initial_state)

    if isinstance(final_state, AgentState):
        data = final_state.model_dump()
    elif isinstance(final_state, dict):
        data = final_state
    else:
        return {
            "final_answer": str(final_state),
            "steps": [],
            "metrics": {},
        }

    final_answer = str(data.get("final_answer") or "")

    raw_steps = data.get("steps") or []
    if isinstance(raw_steps, list):
        steps = [str(s) for s in raw_steps]
    else:
        steps = [str(raw_steps)]

    metrics = data.get("metrics") or {}

    return {
        "final_answer": final_answer,
        "steps": steps,
        "metrics": metrics,
    }
