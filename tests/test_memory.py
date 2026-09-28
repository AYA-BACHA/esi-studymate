"""Test 5: Conversation memory, context resolution, and history summarization."""

import pytest
from app.memory.conversation import ConversationMemoryManager


def test_short_term_memory_recording():
    """Verify messages are recorded in order."""
    mem = ConversationMemoryManager(max_recent=4)
    mem.clear()

    mem.add_message(role="user", content="What is paging?")
    mem.add_message(role="assistant", content="Paging is a memory management scheme.")

    msgs = mem.get_messages()
    assert len(msgs) == 2
    assert msgs[0].content == "What is paging?"
    assert msgs[1].content == "Paging is a memory management scheme."


def test_pronoun_reference_resolution():
    """Verify follow-up queries with 'it' or 'example' are contextualized."""
    mem = ConversationMemoryManager(max_recent=4)
    mem.clear()

    mem.add_message(role="user", content="What is an interrupt?")
    mem.add_message(role="assistant", content="An interrupt is a signal sent to the CPU.")

    # Student asks: 'Give me an example'
    resolved = mem.contextualize_query("Give me an example")
    assert "in reference to 'What is an interrupt?'" in resolved

    # Student asks: 'Explain it simply'
    resolved_it = mem.contextualize_query("Explain it simply")
    assert "in reference to 'What is an interrupt?'" in resolved_it


def test_automatic_summarization_threshold():
    """Verify that exceeding max_recent triggers summarization without crashing."""
    mem = ConversationMemoryManager(max_recent=3)
    mem.clear()

    # Add 6 turns (3 pairs)
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        mem.add_message(role=role, content=f"Message turn number {i}")

    # Should retain only max_recent (3) messages
    assert len(mem.get_messages()) == 3
    # Summary should be generated
    assert len(mem.summary) > 0
