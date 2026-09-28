"""ESI StudyMate Agent.

A simple AI study assistant built using LangChain's create_agent,
with Google Gemini (ChatGoogleGenerativeAI), tools, and short-term memory.
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
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
    """Create and return the StudyMate agent powered by Google Gemini."""
    # Read Google API key
    google_key = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()
    if google_key.startswith("GOOGLE_API_KEY="):
        google_key = google_key.split("GOOGLE_API_KEY=", 1)[1].strip()

    if not google_key:
        google_key = "placeholder-key"

    # Gemini model (defaults to gemini-3.1-flash-lite or custom from .env)
    model_name = os.getenv("GEMINI_MODEL", os.getenv("LLM_MODEL", "gemini-3.1-flash-lite"))

    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_key,
        temperature=0,
    )

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    return agent
