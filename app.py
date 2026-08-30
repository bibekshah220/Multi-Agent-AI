import time
from datetime import datetime

import streamlit as st
from pipeline import (
    _extract_text,
    _run_agent,
    build_reader_agent,
    build_agent,
    writer_chain,
    critic_chain,
)

st.set_page_config(
    page_title="Research Agent Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling --
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; }
    .step-card {
        background: #161b22;
        border: 1px solid #262d38;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .step-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 0.25rem;
    }
    .status-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .pill-done { background: #1f3d2b; color: #4ade80; }
    .pill-active { background: #3d2f1f; color: #fbbf24; }
    .pill-pending { background: #1c2129; color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

STEPS = ["Research", "Reading", "Writing", "Critique"]

if "history" not in st.session_state:
    st.session_state.history = []
if "state" not in st.session_state:
    st.session_state.state = None

# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    st.markdown("### 🧠 Research Agent Studio")
    st.caption("Multi-agent pipeline: Research → Read → Write → Critique")
    st.divider()

    topic = st.text_area(
        "Research topic",
        placeholder="e.g. The latest developments in multi-agent AI systems",
        height=90,
    )
    run = st.button("▶ Run Pipeline", type="primary", use_container_width=True, disabled=not topic)

    st.divider()
    st.markdown("#### Session history")
    if not st.session_state.history:
        st.caption("No runs yet.")
    else:
        for i, h in enumerate(reversed(st.session_state.history)):
            if st.button(f"📄 {h['topic'][:40]}", key=f"hist_{i}", use_container_width=True):
                st.session_state.state = h

# ------------------------------------------------------------------- header --
st.title("Multi-Agent Research Pipeline")
st.caption("Autonomous research, synthesis, and quality review — orchestrated end to end.")

progress_slot = st.empty()


def render_progress(active_idx: int, done_idx: int = -1):
    cols = progress_slot.columns(len(STEPS))
    for i, (col, name) in enumerate(zip(cols, STEPS)):
        if i <= done_idx:
            pill, cls = "done", "pill-done"
        elif i == active_idx:
            pill, cls = "running", "pill-active"
        else:
            pill, cls = "pending", "pill-pending"
        col.markdown(
            f"<div class='step-card'><div class='step-title'>Step {i+1} · {name}</div>"
            f"<span class='status-pill {cls}'>{pill}</span></div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------- run --
if run:
    t0 = time.time()
    state = {"topic": topic}
    render_progress(active_idx=0)

    try:
        with st.spinner("Researching..."):
            research_agent = build_agent()
            state["research"] = _run_agent(
                research_agent,
                f"Research this topic thoroughly and collect key facts with source URLs: {topic}",
            )
        render_progress(active_idx=1, done_idx=0)

        with st.spinner("Reading sources in depth..."):
            reader_agent = build_reader_agent()
            state["reading"] = _run_agent(
                reader_agent,
                "Read the most relevant URLs from the research below and extract additional "
                f"detail and context.\n\nResearch:\n{state['research']}",
            )
        render_progress(active_idx=2, done_idx=1)

        combined_research = f"{state['research']}\n\nDeeper reading:\n{state['reading']}"

        with st.spinner("Writing report..."):
            state["report"] = _extract_text(
                writer_chain.invoke({"topic": topic, "research": combined_research})
            )
        render_progress(active_idx=3, done_idx=2)

        with st.spinner("Critiquing report..."):
            state["critique"] = _extract_text(
                critic_chain.invoke({"topic": topic, "research": combined_research})
            )
        render_progress(active_idx=4, done_idx=3)

        state["elapsed"] = time.time() - t0
        state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.state = state
        st.session_state.history.append(state)

    except Exception as e:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            st.error(
                "Gemini API quota exceeded (free tier allows only ~20 requests/day). "
                "Wait for the quota to reset, use a different API key, or enable billing "
                "on your Google AI Studio project."
            )
        else:
            st.error(f"Pipeline failed: {e}")
        st.stop()

# --------------------------------------------------------------- results --
state = st.session_state.state
if state:
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Report length", f"{len(state.get('report', '').split())} words")
    m2.metric("Runtime", f"{state.get('elapsed', 0):.1f}s" if "elapsed" in state else "—")
    m3.metric("Generated", state.get("timestamp", "—"))

    tab_report, tab_critique, tab_research, tab_reading = st.tabs(
        ["📝 Report", "🔍 Critique", "🌐 Research", "📖 Deep Reading"]
    )
    with tab_report:
        st.markdown(state.get("report", "_No report generated._"))
        st.download_button(
            "Download report (.md)",
            data=state.get("report", ""),
            file_name=f"{state['topic'].strip().replace(' ', '_')}_report.md",
            mime="text/markdown",
        )
    with tab_critique:
        st.markdown(state.get("critique", "_No critique generated._"))
    with tab_research:
        st.markdown(state.get("research", "_No research output._"))
    with tab_reading:
        st.markdown(state.get("reading", "_No reading output._"))
else:
    render_progress(active_idx=-1)
    st.info("Enter a topic in the sidebar and click **Run Pipeline** to begin.")