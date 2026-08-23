from agents import build_reader_agent, build_agent, writer_chain, critic_chain


def _extract_text(content) -> str:
    """Gemini may return content as a list of blocks; normalize to plain text."""
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return content

def _run_agent(agent, prompt: str) -> str:
    result = agent.invoke({"messages": [("user", prompt)]})
    return _extract_text(result["messages"][-1].content)


def run_reserch_pipeline(topic: str) -> dict:
    """Supervisor: runs the research -> reader -> writer -> critic pipeline."""
    state = {"topic": topic}

    # Step 1: Research agent gathers information from the web.
    print("\n=== STEP 1: RESEARCH AGENT ===")
    research_agent = build_agent()
    state["research"] = _run_agent(
        research_agent,
        f"Research this topic thoroughly and collect key facts with source URLs: {topic}",
    )
    print(state["research"])

    