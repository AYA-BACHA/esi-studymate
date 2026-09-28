"""Tests for RAG pipeline (document loading, chunking, vector search)."""

import pytest
from rag import load_and_index_document, search_documents


def test_index_and_search_document():
    """Test loading the sample notes and searching for a topic."""
    # Index sample notes
    num_chunks = load_and_index_document("data/Operating_Systems_Notes.txt")
    assert num_chunks > 0

    # Search for a topic present in the document
    result = search_documents("What is an interrupt?")
    assert "interrupt" in result.lower()
    assert "Source:" in result

    # Search for an absent topic
    absent_result = search_documents("Quantum computing entanglement algorithms")
    # Should either return empty message or low relevance
    assert isinstance(absent_result, str)
