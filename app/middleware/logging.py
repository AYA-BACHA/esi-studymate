"""Observability middleware for logging agent execution traces and measuring latency."""

import time
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Setup standard logger
logger = logging.getLogger("esi_studymate.observability")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [OBSERVER] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AgentEvent(BaseModel):
    """Observable event record during agent graph execution."""
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    event_type: str = Field(description="'start', 'routing', 'tool_call', 'agent_exec', 'approval', 'complete'")
    name: str = Field(description="Name of agent or tool")
    duration_sec: Optional[float] = None
    display_message: str = Field(description="Safe, user-facing summary of the event")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ObservabilityMiddleware:
    """Interception and trace recorder for agent operations."""

    def __init__(self):
        self._events: List[AgentEvent] = []

    def record_event(
        self,
        event_type: str,
        name: str,
        display_message: str,
        duration_sec: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentEvent:
        """Log an observable event without leaking sensitive credentials or chain-of-thought."""
        # Sanitize metadata to remove any secret keys
        safe_meta = {}
        if metadata:
            for k, v in metadata.items():
                if any(secret in k.lower() for secret in ["key", "secret", "password", "token"]):
                    safe_meta[k] = "[REDACTED]"
                else:
                    safe_meta[k] = str(v)[:200]

        event = AgentEvent(
            event_type=event_type,
            name=name,
            duration_sec=round(duration_sec, 3) if duration_sec is not None else None,
            display_message=display_message,
            metadata=safe_meta,
        )
        self._events.append(event)
        
        # Log to standard output
        dur_str = f" in {duration_sec:.3f}s" if duration_sec is not None else ""
        logger.info(f"{display_message}{dur_str}")
        return event

    @contextmanager
    def time_block(self, event_type: str, name: str, display_message: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to measure and log execution duration."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.record_event(
                event_type=event_type,
                name=name,
                display_message=display_message,
                duration_sec=elapsed,
                metadata=metadata,
            )

    def get_events(self) -> List[AgentEvent]:
        """Return all recorded events for the current session."""
        return self._events

    def get_latest_run_events(self, limit: int = 10) -> List[AgentEvent]:
        """Return the most recent run events."""
        return self._events[-limit:]

    def clear(self) -> None:
        """Reset logged events."""
        self._events = []


# Global observer instance
observer = ObservabilityMiddleware()
