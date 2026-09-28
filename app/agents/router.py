"""Router / Supervisor and LangGraph workflow orchestration for ESI StudyMate."""

import json
import time
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.config import settings
from app.agents.tutor import run_tutor_agent
from app.agents.researcher import run_researcher_agent
from app.middleware.logging import observer
from app.llm import llm_client


ROUTER_PROMPT = """You are the Supervisor Router for ESI StudyMate.
Your role is to classify the student's intent and decide which specialized agent(s) should handle it:

1. 'tutor':
   - Explaining general concepts without requiring specific uploaded course slides
   - Answering mathematical questions, calculations, or requesting analogies
   - Beginner or exam-level theoretical explanations

2. 'researcher':
   - Direct requests to search, find, locate, or quote from uploaded PDFs or course documents
   - Questions asking 'What does my course say about...', 'Where is X defined in my PDF', 'Find definition of...'

3. 'both':
   - Requests that require finding specific course slides FIRST, and then tutoring/exercising based on them
   - Example: 'Create 5 exercises about interrupts using my course.'
   - Example: 'Explain how my course defines virtual memory like I am a beginner.'

Respond with a JSON object:
{"agent": "tutor" | "researcher" | "both", "reasoning": "brief explanation"}
"""


def route_intent(query: str) -> Dict[str, str]:
    """Classify user query into agent routing decision."""
    messages = [
        {"role": "system", "content": ROUTER_PROMPT + "\n[ROLE: ROUTE_CLASSIFICATION]"},
        {"role": "user", "content": f"Query: {query}"},
    ]

    try:
        res = llm_client.chat(messages, temperature=0.0)
        content = res.get("content", "").strip()
        # Parse JSON
        if "{" in content and "}" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            data = json.loads(content[start:end])
            decision = data.get("agent", "tutor").lower()
            if decision not in ["tutor", "researcher", "both"]:
                decision = "tutor"
            return {"decision": decision, "reasoning": data.get("reasoning", "Semantic intent classification")}
    except Exception:
        pass

    # Deterministic rule-based fallback
    lower = query.lower()
    if ("exercise" in lower or "quiz" in lower) and any(w in lower for w in ["course", "pdf", "slide", "material"]):
        return {"decision": "both", "reasoning": "Retrieval required before exercise generation"}
    elif any(w in lower for w in ["find", "where is", "page", "in my pdf", "in my course", "search course", "slides"]):
        return {"decision": "researcher", "reasoning": "Document search intent detected"}
    else:
        return {"decision": "tutor", "reasoning": "Conceptual explanation or exercise intent"}


# ---------------- LangGraph Nodes ----------------

def supervisor_router_node(state: AgentState) -> AgentState:
    """Classify user query and update state with routing decision."""
    observer.record_event(
        event_type="start",
        name="ESI StudyMate",
        display_message=f"Agent started with query: '{state.query[:50]}...'",
    )

    with observer.time_block(
        event_type="routing",
        name="Agent Router",
        display_message="Supervisor analyzing student query intent...",
    ):
        routing_info = route_intent(state.query)

    state.route_decision = routing_info["decision"]
    state.route_reasoning = routing_info["reasoning"]

    observer.record_event(
        event_type="routing",
        name="Agent Router",
        display_message=f"Routing decision: [{state.route_decision.upper()}] ({state.route_reasoning})",
        metadata={"decision": state.route_decision, "reasoning": state.route_reasoning},
    )

    # Check Human-In-The-Loop approval requirement
    if settings.require_retrieval_approval and state.route_decision in ["researcher", "both"]:
        if state.human_approved is None:
            state.requires_human_approval = True
            state.approval_prompt = (
                f"Agent wants to search uploaded course documents for: '{state.query}'. "
                "Do you want to approve this search?"
            )
            observer.record_event(
                event_type="approval",
                name="Human-In-The-Loop",
                display_message="Waiting for student approval to perform course retrieval...",
            )

    return state


def conditional_edge_supervisor(state: AgentState) -> str:
    """Determine the next step after routing and approval check."""
    # If human approval is required and pending, halt for user interaction
    if state.requires_human_approval and state.human_approved is None:
        return "await_approval"

    # If human rejected search, fallback to tutor general response
    if state.human_approved is False:
        return "tutor"

    if state.route_decision == "researcher":
        return "researcher"
    elif state.route_decision == "both":
        return "researcher_then_tutor"
    else:
        return "tutor"


def build_agent_graph():
    """Construct and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", supervisor_router_node)
    workflow.add_node("researcher", run_researcher_agent)
    workflow.add_node("tutor", run_tutor_agent)

    # Entry point
    workflow.set_entry_point("router")

    # Conditional routing edge
    workflow.add_conditional_edges(
        "router",
        conditional_edge_supervisor,
        {
            "researcher": "researcher",
            "tutor": "tutor",
            "researcher_then_tutor": "researcher",
            "await_approval": END,
        },
    )

    # Edge from researcher: if route was 'both', proceed to tutor; else end
    def after_researcher_edge(state: AgentState) -> str:
        if state.route_decision == "both":
            return "tutor"
        return "end"

    workflow.add_conditional_edges(
        "researcher",
        after_researcher_edge,
        {
            "tutor": "tutor",
            "end": END,
        },
    )

    workflow.add_edge("tutor", END)

    return workflow.compile()


# Compiled agent graph singleton
agent_graph = build_agent_graph()


def execute_agent(
    query: str,
    teaching_style: str = "standard",
    human_approved: Any = None,
    existing_messages: Any = None,
    summary: str = "",
) -> AgentState:
    """Main execution function invoking the compiled agent graph."""
    initial_state = AgentState(
        query=query,
        teaching_style=teaching_style,
        human_approved=human_approved,
        messages=existing_messages or [],
        summary=summary,
    )

    result_state_dict = agent_graph.invoke(initial_state)
    if isinstance(result_state_dict, dict):
        result_state = AgentState(**result_state_dict)
    else:
        result_state = result_state_dict

    # Final observability event
    if not result_state.requires_human_approval or result_state.human_approved is not None:
        observer.record_event(
            event_type="complete",
            name="ESI StudyMate",
            display_message="Agent run completed successfully.",
        )

    return result_state
