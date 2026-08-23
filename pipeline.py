from agents import build_reader_agent, build_agent, writer_chain, critic_chain


def _extract_text(content) -> str:
    """Gemini may return content as a list of blocks; normalize to plain text."""
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return content


