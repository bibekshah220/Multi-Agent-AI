from agents import build_reader_agent, build_agent, writer_chain, critic_chain


def _extract_text(content) -> str:
    """Gemini may return content as a list of blocks; normalize to plain text."""
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return content

def _run_agent(agent, prompt: str) -> str:
    # recursion_limit caps tool-call iterations so the agent can't loop forever.
    result = agent.invoke(
        {"messages": [("user", prompt)]},
        config={"recursion_limit": 12},
    )
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

    # Step 2: Reader agent reads sources in depth to enrich the research.
    print("\n=== STEP 2: READER AGENT ===")
    reader_agent = build_reader_agent()
    state["reading"] = _run_agent(
        reader_agent,
        "Read the most relevant URLs from the research below and extract additional "
        f"detail and context.\n\nResearch:\n{state['research']}",
    )
    print(state["reading"])

    # Combine gathered material for the writer and critic.
    combined_research = f"{state['research']}\n\nDeeper reading:\n{state['reading']}"

    # Step 3: Writer chain produces the final report.
    print("\n=== STEP 3: WRITER CHAIN ===")
    state["report"] = writer_chain.invoke(
        {"topic": topic, "research": combined_research}
    )
    print(state["report"])

    # Step 4: Critic chain evaluates and scores the report.
    print("\n=== STEP 4: CRITIC CHAIN ===")
    state["critique"] = critic_chain.invoke(
        {"topic": topic, "research": combined_research}
    )
    print(state["critique"])

    return state


if __name__ == "__main__":
    run_reserch_pipeline("The latest developments in multi-agent AI systems")