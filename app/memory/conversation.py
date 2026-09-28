"""Conversation memory manager with short-term state and automatic history summarization."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from app.config import settings
from app.state import Message, SourceCitation
from app.llm import llm_client


class ConversationMemoryManager:
    """Maintains active conversation turns and compacts older history."""

    def __init__(self, max_recent: int = 6):
        self.max_recent = max_recent
        self.messages: List[Message] = []
        self.summary: str = ""

    def add_message(
        self,
        role: str,
        content: str,
        sources: Optional[List[SourceCitation]] = None,
        agent_name: Optional[str] = None,
    ) -> Message:
        """Append a message to the active history."""
        msg = Message(
            role=role,
            content=content,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            sources=sources or [],
            agent_name=agent_name,
        )
        self.messages.append(msg)
        self._check_and_summarize()
        return msg

    def get_messages(self) -> List[Message]:
        """Return the current active message list."""
        return self.messages

    def clear(self) -> None:
        """Reset conversation history and summary."""
        self.messages = []
        self.summary = ""

    def _check_and_summarize(self) -> None:
        """Compress older turns when history exceeds maximum window."""
        if len(self.messages) <= self.max_recent:
            return

        # Keep the most recent messages, summarize the older ones
        cutoff = len(self.messages) - self.max_recent
        older_turns = self.messages[:cutoff]
        self.messages = self.messages[cutoff:]

        # Format older conversation text
        older_dialogue = []
        for turn in older_turns:
            older_dialogue.append(f"{turn.role.capitalize()}: {turn.content}")
        text_to_summarize = "\n".join(older_dialogue)

        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a concise conversation summarizer. Update the existing running summary "
                    "with the key topics, concepts, and student questions discussed in these older messages. "
                    "Keep the summary under 150 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous summary: '{self.summary}'\n\n"
                    f"New messages to incorporate:\n{text_to_summarize}"
                ),
            },
        ]

        try:
            res = llm_client.chat(summary_prompt, temperature=0.1)
            self.summary = res.get("content", "").strip() or self.summary
        except Exception:
            # Deterministic local summary fallback
            topics = [turn.content[:40] for turn in older_turns if turn.role == "user"]
            self.summary = f"Student previously asked about: {', '.join(topics)}."

    def contextualize_query(self, query: str) -> str:
        """Resolve pronouns or follow-up references using recent conversation context.
        
        Example:
        Turn 1: 'What is an interrupt?'
        Turn 2: 'Give me an example' -> resolves context to 'Give me an example of an interrupt'
        """
        lower_q = query.lower().strip()
        followup_signals = ["it", "this", "that", "them", "give me an example", "another one", "why", "how", "more details"]
        
        needs_context = any(
            lower_q == sig or lower_q.startswith(sig + " ") or f" {sig} " in lower_q
            for sig in followup_signals
        )

        if not needs_context or not self.messages:
            return query

        # Find the last topic discussed from previous user messages
        last_user_msg = ""
        for m in reversed(self.messages):
            if m.role == "user":
                last_user_msg = m.content
                break

        if last_user_msg:
            return f"{query} (context: in reference to '{last_user_msg}')"
        return query


# Global memory manager
conversation_memory = ConversationMemoryManager(max_recent=settings.max_short_term_messages)
