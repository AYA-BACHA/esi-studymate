"""ESI StudyMate Agent.

A simple AI study assistant built using LangChain's create_agent,
with tools and short-term conversation memory.
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from tools import search_course_material, calculator

load_dotenv()

SYSTEM_PROMPT = """You are StudyMate, an AI study assistant for university students.

Help the student understand their course material.

When the student asks about course topics, concepts, or definitions, use the search_course_material tool.

Base your answers on the retrieved course content when possible.

If the answer is not present in the provided course material, clearly tell the student that the uploaded documents do not contain enough information.

When mathematical calculations are needed, use the calculator tool.
"""

# Short-term memory checkpointer
memory = MemorySaver()

# Tools available to the agent
tools = [search_course_material, calculator]


def get_agent():
    """Create and return the StudyMate agent."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        api_key = "placeholder-key"

    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=api_key,
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    return agent
