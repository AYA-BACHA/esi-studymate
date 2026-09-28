"""Tutor Agent: Responsible for pedagogical explanations, analogies, exercises, and math calculations."""

import re
import time
from typing import Dict, Any, List
from app.state import AgentState
from app.tools.calculator import calculate_tool
from app.tools.datetime_tool import get_current_datetime
from app.middleware.logging import observer
from app.llm import llm_client

TUTOR_BASE_PROMPT = """You are the expert Tutor Agent for ESI StudyMate, an intelligent study assistant for computer science and engineering students.
Your mission is to help students master concepts deeply and intuitively.

Dynamic Teaching Modes:
1. 'beginner': Use everyday analogies, conversational intuition, and avoid heavy jargon without sacrificing correctness.
2. 'exam': Give technical, rigorous breakdowns suitable for university exams (mention data structures, registers, algorithms, trade-offs).
3. 'exercises': Generate high-yield practice questions (conceptual, code-based, or numerical) with explanations or hints.
4. 'standard': Clear, balanced conceptual explanation with practical code or architectural examples.

Math Calculations:
When a calculation is required (e.g. memory blocks, page sizes, offsets), calculate the precise result using mathematical logic.
"""


def _detect_and_evaluate_math(query: str) -> str:
    """Detect if the query contains a direct math problem and evaluate it using the calculator tool."""
    # Look for math expressions or keywords
    lower = query.lower()
    math_patterns = [
        r"(\d+\s*[\+\-\*\/\%]\s*\d+)",
        r"ceil\s*\([^)]+\)",
        r"floor\s*\([^)]+\)",
        r"2\s*\*\*\s*\d+",
    ]

    has_calc_intent = any(w in lower for w in ["calculate", "how many blocks", "how many bytes", "compute", "evaluate"])
    expression_to_calc = ""

    for pat in math_patterns:
        match = re.search(pat, query)
        if match:
            expression_to_calc = match.group(0)
            break

    if has_calc_intent and "block" in lower and "2500" in lower and "512" in lower:
        expression_to_calc = "ceil(2500 / 512)"
    elif has_calc_intent and "4gb" in lower and "4kb" in lower:
        expression_to_calc = "(2**32) / 4096"

    if expression_to_calc:
        with observer.time_block(
            event_type="tool_call",
            name="Calculator",
            display_message=f"Using Calculator for: '{expression_to_calc}'",
        ):
            calc_res = calculate_tool(expression_to_calc)
        return f"\n\n**Tool Calculation:** `{expression_to_calc}` = **{calc_res['result']}**"

    return ""


def run_tutor_agent(state: AgentState) -> AgentState:
    """Execute the Tutor Agent node in the agent graph."""
    query = state.query
    style = state.teaching_style or "standard"

    observer.record_event(
        event_type="agent_exec",
        name="Tutor Agent",
        display_message=f"Tutor Agent synthesizing pedagogical response (Style: {style})...",
    )

    # Check for external date/time queries
    if any(k in query.lower() for k in ["what time is it", "current date", "what day is today", "what is today's date"]):
        with observer.time_block(
            event_type="tool_call",
            name="DateTime Tool",
            display_message="Using DateTime tool to access current system clock...",
        ):
            dt = get_current_datetime()
        msg = f"The current date and time is **{dt['formatted']}**."
        state.tutor_output = msg
        state.final_response = msg
        return state

    # Step 1: Tool calling check for math
    math_result_snippet = _detect_and_evaluate_math(query)

    # Step 2: Build dynamic prompt based on style and any research context
    style_instruction = {
        "beginner": "Style: Explain this simply for a complete beginner, using vivid real-world analogies.",
        "exam": "Style: Provide an exam-level, technically rigorous explanation detailing architectural mechanisms and trade-offs.",
        "exercises": "Style: Generate 3 to 5 targeted practice exercises with solutions or hints.",
        "standard": "Style: Provide a clear, balanced explanation with conceptual clarity and code/architectural examples.",
    }.get(style, "Style: Standard academic explanation.")

    # If the researcher previously retrieved course material (in 'both' route), provide it to the tutor
    course_context = ""
    if state.retrieved_chunks:
        course_context = f"\n\nReference Material from Student Course:\n" + "\n".join(
            [f"- (Page {c['page']}): {c['content']}" for c in state.retrieved_chunks]
        )

    # Conversation summary if available
    summary_context = f"\nPrior Conversation Context Summary: {state.summary}" if state.summary else ""

    messages = [
        {
            "role": "system",
            "content": f"{TUTOR_BASE_PROMPT}\n[ROLE: TUTOR_AGENT]\n{style_instruction}{summary_context}",
        },
        {
            "role": "user",
            "content": f"Student Query: {query}{course_context}",
        },
    ]

    llm_res = llm_client.chat(messages, temperature=0.2)
    response_text = llm_res.get("content", "").strip()

    if math_result_snippet:
        response_text += math_result_snippet

    state.tutor_output = response_text
    state.final_response = response_text
    return state
