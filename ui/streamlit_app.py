"""ESI StudyMate - Enterprise-Grade Streamlit User Interface.

A clean, professional academic assistant interface following enterprise design system
standards: crisp typography, neutral slate color scheme, subtle borders, and zero decorative emojis.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from app.config import settings
from app.agents.router import execute_agent
from app.memory.conversation import conversation_memory
from app.rag.indexer import document_indexer
from app.rag.vectorstore import course_vector_store
from app.middleware.logging import observer
from app.state import SourceCitation

# ----------------- Streamlit Page Configuration -----------------
st.set_page_config(
    page_title="ESI StudyMate",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- Enterprise Design System (Tailwind-Inspired) -----------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

code, pre, .font-mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Remove default excess padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* Header typography */
h1 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    color: #f8fafc !important;
    margin-bottom: 0.375rem !important;
}

h2 {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
    color: #f1f5f9 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

/* Subdued enterprise badge */
.badge-tag {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    font-size: 0.6875rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-radius: 4px;
    background: #1e293b;
    border: 1px solid #334155;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}

/* Refined Trace Cards (No colored left border bar, clean slate-800 borders) */
.trace-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 0.625rem 0.75rem;
    margin-bottom: 0.5rem;
}

.trace-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
}

.trace-component {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.trace-duration {
    font-size: 0.6875rem;
    font-family: 'JetBrains Mono', monospace;
    color: #64748b;
}

.trace-message {
    font-size: 0.8125rem;
    color: #cbd5e1;
    line-height: 1.4;
}

/* Source Citation Card */
.source-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 0.75rem;
    margin-top: 0.5rem;
}

.source-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.8125rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
}

.source-score {
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #64748b;
}

.source-preview {
    font-size: 0.75rem;
    color: #94a3b8;
    line-height: 1.45;
}

/* Approval Card */
.approval-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
}

.approval-title {
    font-size: 0.9375rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 0.375rem;
}

.approval-query {
    font-size: 0.875rem;
    color: #cbd5e1;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
    margin: 0.625rem 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------- Session State Initialization -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

if "teaching_style" not in st.session_state:
    st.session_state.teaching_style = "standard"


# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown('<span class="badge-tag">Academic Architecture</span>', unsafe_allow_html=True)
    st.markdown("### ESI StudyMate")
    st.caption("AI-Assisted Study & Retrieval System")

    st.markdown("---")
    st.markdown("#### Document Management")

    # Document Uploader
    uploaded_files = st.file_uploader(
        "Upload Course Material",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload lecture notes, textbooks, or syllabus materials.",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            key = f"indexed_{uploaded_file.name}_{len(file_bytes)}"
            if key not in st.session_state:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    result = document_indexer.index_bytes(file_bytes, uploaded_file.name)
                    st.session_state[key] = True
                    st.success(f"Indexed {result['file_name']} ({result['indexed_chunks']} chunks)")

    stats = course_vector_store.get_stats()
    st.metric("Indexed Chunks", stats["total_chunks"])

    if stats["total_chunks"] == 0:
        if st.button("Load Standard OS Notes", use_container_width=True):
            from data.sample_data_loader import create_sample_documents
            ch3, ch4 = create_sample_documents()
            document_indexer.index_file(ch3)
            document_indexer.index_file(ch4)
            st.success("Loaded 2 course chapters into vector index.")
            st.rerun()

    st.markdown("---")
    st.markdown("#### Configuration")

    # Dynamic Teaching Style Selector
    style_options = {
        "standard": "Balanced Academic",
        "beginner": "Conceptual Foundations",
        "exam": "Technical / Exam-Level",
        "exercises": "Exercise Generation",
    }
    selected_style_key = st.selectbox(
        "Teaching Mode",
        options=list(style_options.keys()),
        format_func=lambda k: style_options[k],
        index=list(style_options.keys()).index(st.session_state.teaching_style),
        help="Runtime instruction modification without separate code paths.",
    )
    st.session_state.teaching_style = selected_style_key

    # Human-in-the-loop toggle
    st.toggle(
        "Require Retrieval Approval",
        value=settings.require_retrieval_approval,
        key="hitl_toggle",
        help="Pauses execution prior to vector store queries to await operator confirmation.",
        on_change=lambda: setattr(settings, "require_retrieval_approval", st.session_state.hitl_toggle),
    )

    # Clear conversation button
    if st.button("Reset Session", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pending_approval = None
        conversation_memory.clear()
        observer.clear()
        st.rerun()


# ----------------- Main UI Layout -----------------
col_main, col_observability = st.columns([7, 3])

with col_main:
    st.markdown('<span class="badge-tag">National Higher School of Computer Science (ESI)</span>', unsafe_allow_html=True)
    st.title("ESI StudyMate")
    st.markdown(
        "Autonomous academic study system supporting multi-agent intent routing, "
        "page-cited document retrieval (RAG), and deterministic arithmetic evaluation."
    )

    # Prompt Template Actions
    st.caption("Common queries:")
    chip_cols = st.columns(4)
    with chip_cols[0]:
        if st.button("Explain Interrupts", use_container_width=True):
            st.session_state.selected_prompt = "Explain interrupts from my Operating Systems course."
    with chip_cols[1]:
        if st.button("Recursion (Foundations)", use_container_width=True):
            st.session_state.selected_prompt = "Explain recursion like I am a beginner."
    with chip_cols[2]:
        if st.button("Calculate Block Allocation", use_container_width=True):
            st.session_state.selected_prompt = "Calculate how many memory blocks are required for 2500 bytes with 512-byte blocks."
    with chip_cols[3]:
        if st.button("Generate Exercises", use_container_width=True):
            st.session_state.selected_prompt = "Create 5 exercises about interrupts using my course."

    # ----------------- Human-In-The-Loop Approval Card -----------------
    if st.session_state.pending_approval:
        pending = st.session_state.pending_approval
        st.markdown(
            f"""
            <div class="approval-card">
                <div class="approval-title">Action Approval Required</div>
                <div style="font-size: 0.8125rem; color: #94a3b8;">
                    The supervisor router has requested permission to query course documents for:
                </div>
                <div class="approval-query">{pending['query']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_app1, col_app2 = st.columns(2)
        with col_app1:
            if st.button("Approve Retrieval", use_container_width=True, type="primary"):
                st.session_state.pending_approval = None
                with st.spinner("Executing retrieval..."):
                    res_state = execute_agent(
                        query=pending["query"],
                        teaching_style=pending["teaching_style"],
                        human_approved=True,
                        existing_messages=conversation_memory.get_messages(),
                        summary=conversation_memory.summary,
                    )
                    conversation_memory.add_message(role="assistant", content=res_state.final_response, sources=res_state.sources)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": res_state.final_response,
                        "sources": res_state.sources,
                        "route": res_state.route_decision,
                    })
                st.rerun()

        with col_app2:
            if st.button("Decline (General Knowledge Only)", use_container_width=True):
                st.session_state.pending_approval = None
                with st.spinner("Processing without document search..."):
                    res_state = execute_agent(
                        query=pending["query"],
                        teaching_style=pending["teaching_style"],
                        human_approved=False,
                        existing_messages=conversation_memory.get_messages(),
                        summary=conversation_memory.summary,
                    )
                    conversation_memory.add_message(role="assistant", content=res_state.final_response)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": res_state.final_response,
                        "sources": [],
                        "route": "tutor",
                    })
                st.rerun()

    # ----------------- Chat History Display -----------------
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Source Citations
            if message.get("sources"):
                with st.expander(f"Sources ({len(message['sources'])})", expanded=False):
                    seen = set()
                    for src in message["sources"]:
                        key = f"{src.source}_{src.page}"
                        if key not in seen:
                            seen.add(key)
                            score_label = f"Score: {src.score:.2f}" if src.score else ""
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <div class="source-header">
                                        <span>{src.source} — Page {src.page}</span>
                                        <span class="source-score">{score_label}</span>
                                    </div>
                                    <div class="source-preview">{src.preview}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # ----------------- Chat Input Handling -----------------
    default_prompt = st.session_state.pop("selected_prompt", None)
    user_input = st.chat_input("Enter a query, calculation, or exercise request...")

    prompt_to_run = user_input or default_prompt

    if prompt_to_run and not st.session_state.pending_approval:
        contextual_query = conversation_memory.contextualize_query(prompt_to_run)

        conversation_memory.add_message(role="user", content=prompt_to_run)
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_run})

        with st.chat_message("user"):
            st.markdown(prompt_to_run)

        with st.chat_message("assistant"):
            with st.spinner("Processing query..."):
                res_state = execute_agent(
                    query=contextual_query,
                    teaching_style=st.session_state.teaching_style,
                    existing_messages=conversation_memory.get_messages(),
                    summary=conversation_memory.summary,
                )

            if res_state.requires_human_approval and res_state.human_approved is None:
                st.session_state.pending_approval = {
                    "query": contextual_query,
                    "teaching_style": st.session_state.teaching_style,
                }
                st.rerun()

            st.markdown(res_state.final_response)

            if res_state.sources:
                with st.expander(f"Sources ({len(res_state.sources)})", expanded=False):
                    seen = set()
                    for src in res_state.sources:
                        key = f"{src.source}_{src.page}"
                        if key not in seen:
                            seen.add(key)
                            score_label = f"Score: {src.score:.2f}" if src.score else ""
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <div class="source-header">
                                        <span>{src.source} — Page {src.page}</span>
                                        <span class="source-score">{score_label}</span>
                                    </div>
                                    <div class="source-preview">{src.preview}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            conversation_memory.add_message(
                role="assistant",
                content=res_state.final_response,
                sources=res_state.sources,
            )
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": res_state.final_response,
                "sources": res_state.sources,
                "route": res_state.route_decision,
            })

        st.rerun()


# ----------------- Observability Activity Feed -----------------
with col_observability:
    st.markdown("#### Execution Trace")
    st.caption("Observable events and component latency")

    events = observer.get_latest_run_events(limit=12)

    if not events:
        st.info("System idle. Traces will appear here during query execution.")
    else:
        for ev in reversed(events):
            dur_text = f"{ev.duration_sec:.3f}s" if ev.duration_sec is not None else ""
            st.markdown(
                f"""
                <div class="trace-card">
                    <div class="trace-header">
                        <span class="trace-component">{ev.name}</span>
                        <span class="trace-duration">{dur_text}</span>
                    </div>
                    <div class="trace-message">{ev.display_message}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if conversation_memory.summary:
        with st.expander("Context Summary", expanded=False):
            st.caption(conversation_memory.summary)
