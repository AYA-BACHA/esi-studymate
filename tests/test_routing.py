"""Test 6: Multi-Agent Supervisor Routing."""

import pytest
from app.agents.router import route_intent, execute_agent


def test_tutor_routing():
    """Verify conceptual, simple explanation, or math queries route to Tutor."""
    res1 = route_intent("Explain recursion simply.")
    assert res1["decision"] == "tutor"

    res2 = route_intent("Calculate how many blocks are required for 2500 bytes.")
    assert res2["decision"] == "tutor"


def test_researcher_routing():
    """Verify document search requests route to Research Agent."""
    res1 = route_intent("Find the definition of virtual memory in my course.")
    assert res1["decision"] == "researcher"

    res2 = route_intent("Where is interrupt vector table defined in my PDF?")
    assert res2["decision"] == "researcher"


def test_both_routing_coordination():
    """Verify composite requests requiring search + exercise generation route to Both."""
    res = route_intent("Create 5 exercises about interrupts using my course.")
    assert res["decision"] == "both"


def test_graph_execution_routing():
    """Verify end-to-end graph execution state matches routing intent."""
    state = execute_agent("Explain recursion like I am a beginner.", teaching_style="beginner")
    assert state.route_decision == "tutor"
    assert state.final_response != ""
