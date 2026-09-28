"""Tests for agent creation and tool binding."""

import pytest
from agent import get_agent, tools


def test_agent_initialization():
    """Verify agent can be created with tools and checkpointer."""
    agent = get_agent()
    assert agent is not None


def test_tools_list():
    """Verify tools list contains search and calculator."""
    tool_names = [t.name for t in tools]
    assert "search_course_material" in tool_names
    assert "calculator" in tool_names
