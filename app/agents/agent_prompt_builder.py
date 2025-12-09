# app/agents/agent_prompt_builder.py

def build_agent_final_prompt(user_query: str, tool_outputs: dict) -> str:
    """
    Build the final summary prompt for the agent.
    """

    tool_summary = ""

    for name, out in tool_outputs.items():
        tool_summary += f"\n[{name.upper()} OUTPUT]\n{str(out)[:800]}\n"

    prompt = f"""
You are an AI agent. Use the tool outputs to answer the question.

If tools do not give enough information, say:
"The tools do not contain enough information."

User question:
{user_query}

Tool results:
{tool_summary}

Give a clear and direct final answer.
"""
    return prompt.strip()
