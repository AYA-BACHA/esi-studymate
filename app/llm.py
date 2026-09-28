"""Unified LLM client interface for ESI StudyMate.

Supports OpenAI, Groq, OpenRouter, Ollama, and a deterministic Mock provider
for automated testing and offline development.
"""

from typing import Any, Dict, List, Optional
import os
import json
import logging
from app.config import settings

logger = logging.getLogger("esi_studymate.llm")


class MockLLMClient:
    """Deterministic offline mock LLM client for testing and offline environments."""

    def __init__(self, model: str = "mock-model"):
        self.model = model

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate response based on message content."""
        last_message = messages[-1]["content"] if messages else ""
        system_content = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""

        # Router classification simulation
        if "ROUTE_CLASSIFICATION" in system_content:
            text = last_message.lower()
            if "exercise" in text or "quiz" in text:
                decision = "both"
            elif any(w in text for w in ["find", "search", "where is", "page", "pdf", "course", "slides", "definition"]):
                decision = "researcher"
            elif any(w in text for w in ["calculate", "how many", "math", "+", "-", "*", "/"]):
                decision = "tutor"
            else:
                decision = "tutor"
            return {
                "role": "assistant",
                "content": json.dumps({"agent": decision, "reasoning": f"Routed based on intent analysis of query: {last_message[:40]}"}),
            }

        # Tutor agent simulation
        if "TUTOR_AGENT" in system_content:
            if "beginner" in last_message.lower() or "simple" in last_message.lower():
                return {
                    "role": "assistant",
                    "content": f"Here is a friendly, beginner-level explanation:\n\nThink of this concept like an everyday situation: when a doorbell rings while you're reading a book, you bookmark your page, answer the door, and then return to your book right where you left off!\n\nIn computer systems, that is exactly how interrupts allow hardware to get the CPU's attention without wasting cycles polling.",
                }
            elif "exercise" in last_message.lower():
                return {
                    "role": "assistant",
                    "content": "### Practice Exercises:\n1. **Question 1**: What is the difference between a software interrupt (trap) and a hardware interrupt?\n2. **Question 2**: Explain why interrupt vector tables are stored in privileged memory.\n3. **Question 3**: If an I/O transfer takes 20ms and interrupt servicing takes 50µs, what is the CPU overhead percentage?",
                }
            elif "exam" in last_message.lower():
                return {
                    "role": "assistant",
                    "content": "### Exam-Level Technical Breakdown:\n- **Mechanism**: Hardware triggers interrupt line → CPU completes current instruction execution cycle → saves PC and flags to kernel stack → loads Interrupt Service Routine (ISR) address from Interrupt Vector Table (IVT).\n- **Key Trade-offs**: Interrupt latency vs polling overhead, interrupt masking, nested priority handling.",
                }
            else:
                return {
                    "role": "assistant",
                    "content": f"Concept Explanation: {last_message}\n\nIn computer architecture and operating systems, this mechanism coordinates asynchronous execution between devices and the CPU to optimize utilization.",
                }

        # Researcher agent simulation
        if "RESEARCHER_AGENT" in system_content:
            return {
                "role": "assistant",
                "content": "Based on the retrieved course documents, the relevant material specifies the formal definitions and architectural mechanisms as cited.",
            }

        # Fallback general response
        return {
            "role": "assistant",
            "content": f"I have processed your query: '{last_message}'. As your ESI StudyMate assistant, I am ready to help you learn.",
        }


class LLMClient:
    """Wrapper supporting standard OpenAI-compatible providers and Mock fallback."""

    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.api_key = settings.api_key
        self.base_url = settings.base_url

        self._openai_client = None
        self._mock_client = MockLLMClient(model=self.model)

        # Initialize OpenAI client if valid key is available
        if self.provider != "mock" and self.api_key and not self.api_key.startswith("your_"):
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._openai_client = OpenAI(**kwargs)
                logger.info(f"Initialized OpenAI-compatible client for provider: {self.provider}, model: {self.model}")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client ({e}), falling back to mock.")
                self._openai_client = None

    def is_live(self) -> bool:
        """Check if an active remote API client is configured."""
        return self._openai_client is not None

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request to the active LLM provider."""
        if self._openai_client:
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if tools:
                    kwargs["tools"] = tools
                response = self._openai_client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                tool_calls = []
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        })
                return {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": tool_calls if tool_calls else None,
                }
            except Exception as e:
                logger.error(f"Error calling live LLM API: {e}. Falling back to deterministic local mock.")
                return self._mock_client.chat_completion(messages, temperature, tools)
        else:
            return self._mock_client.chat_completion(messages, temperature, tools)


# Global instance
llm_client = LLMClient()
