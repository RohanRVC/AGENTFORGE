# app/agents/controller.py

from app.agents.tools import rag_tool, calculator_tool, web_scraper_tool
from app.agents.planner import plan_steps
from app.services.llama_service import run_llama
from app.core.metrics import timer
from app.utils.latency import measure_latency


@measure_latency("Agent Execution")
def run_agent_controller(task: str):

    start_time = timer()

    # AI agent planning
    steps = plan_steps(task)
    trace = []
    metrics = {
        "tool_times": {},
        "total_run_time": 0
    }

    tool_map = {
        "rag": rag_tool,
        "calculator": calculator_tool,
        "web_scraper": web_scraper_tool
    }

    tool_outputs = {}

    for step in steps:
        tool_name = step["tool"]
        tool_input = step["input"]

        if tool_name not in tool_map:
            trace.append(f"Unknown tool: {tool_name}")
            continue

        tool_fn = tool_map[tool_name]

        tool_start = timer()
        try:
            result = tool_fn(tool_input)
            tool_outputs[tool_name] = result
            trace.append(f"{tool_name} → OK")
        except Exception as e:
            result = f"ERROR: {str(e)}"
            trace.append(f"{tool_name} → ERROR: {str(e)}")

        tool_end = timer()
        metrics["tool_times"][tool_name] = round(tool_end - tool_start, 4)

    # --- Final synthesis prompt ---
    final_prompt = f"""
User Task:
{task}

Planner Steps:
{steps}

Tool Results:
{tool_outputs}

Write the final answer in very clear language.
"""

    final_answer, cost_info = run_llama(final_prompt)

    end_time = timer()
    metrics["total_run_time"] = round(end_time - start_time, 4)

    return {
        "final_answer": final_answer,
        "steps": trace,
        "metrics": metrics,
        "llm_cost": cost_info     # <-- ADDED (important for assignment)
    }
