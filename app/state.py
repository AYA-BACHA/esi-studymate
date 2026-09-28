"""State definition for ESI StudyMate LangGraph agents."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Citation metadata for retrieved course documents."""
    source: str = Field(description="Document filename, e.g. Operating_Systems.pdf")
    page: int = Field(default=1, description="Page number of the cited content")
    preview: Optional[str] = Field(default="", description="Short snippet preview")
    score: Optional[float] = Field(default=0.0, description="Similarity score")


class ToolExecutionRecord(BaseModel):
    """Observable record of a tool execution."""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: str
    execution_time_sec: float


class Message(BaseModel):
    """Conversation message with optional citations and metadata."""
    role: str = Field(description="Role: user, assistant, system, or tool")
    content: str = Field(description="Textual content of the message")
    timestamp: Optional[str] = None
    sources: Optional[List[SourceCitation]] = Field(default_factory=list)
    agent_name: Optional[str] = None


class AgentState(BaseModel):
    """Shared state dictionary passed across LangGraph nodes."""

    # User input and conversation
    query: str = Field(default="", description="The latest query from the student")
    teaching_style: str = Field(
        default="standard",
        description="Dynamic style: 'standard', 'beginner', 'exam', 'exercises'"
    )
    messages: List[Message] = Field(default_factory=list, description="Recent conversation history")
    summary: str = Field(default="", description="Rolling summary of older conversation history")

    # Routing
    route_decision: str = Field(default="tutor", description="'tutor', 'researcher', or 'both'")
    route_reasoning: str = Field(default="", description="Explanation of why this route was selected")

    # RAG Context & Citations
    retrieved_chunks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Raw retrieved text chunks with document and page metadata"
    )
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Formatted source citations to be displayed to the student"
    )

    # Human-In-The-Loop
    requires_human_approval: bool = Field(default=False)
    approval_prompt: str = Field(default="")
    human_approved: Optional[bool] = Field(default=None)

    # Tool Execution Observability
    tool_history: List[ToolExecutionRecord] = Field(default_factory=list)

    # Agent Outputs
    researcher_output: str = Field(default="")
    tutor_output: str = Field(default="")
    final_response: str = Field(default="")
