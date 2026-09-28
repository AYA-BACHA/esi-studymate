"""Test 3 & 4: QA with Information Present vs Absent in Documents."""

import pytest
from app.rag.loader import DocumentPage
from app.rag.chunker import RecursiveChunker
from app.rag.vectorstore import CourseVectorStore
from app.rag.retriever import CourseRetriever
from app.state import AgentState
from app.agents.researcher import run_researcher_agent


@pytest.fixture
def populated_vector_store():
    """Create isolated in-memory vector store with sample OS knowledge."""
    store = CourseVectorStore(in_memory=True)
    store.clear()

    page = DocumentPage(
        content=(
            "The Interrupt Vector Table (IVT) contains addresses of Interrupt Service Routines (ISRs) "
            "stored in protected kernel memory. Hardware interrupts are asynchronous."
        ),
        source="Operating_Systems.pdf",
        page_number=12,
    )
    chunker = RecursiveChunker(chunk_size=400, chunk_overlap=50)
    store.add_chunks(chunker.chunk_documents([page]))
    return store


def test_question_with_information_present(populated_vector_store, monkeypatch):
    """Test 3: Information present in document produces grounded answer with valid citations."""
    retriever = CourseRetriever(vector_store=populated_vector_store)
    monkeypatch.setattr("app.agents.researcher.course_retriever", retriever)

    state = AgentState(query="What is stored in the Interrupt Vector Table?")
    result = run_researcher_agent(state)

    # Must be grounded
    assert len(result.retrieved_chunks) > 0
    assert len(result.sources) > 0
    assert result.sources[0].source == "Operating_Systems.pdf"
    assert result.sources[0].page == 12
    # Response must cite sources
    assert "Operating_Systems.pdf" in result.final_response
    assert "page 12" in result.final_response.lower()


def test_question_with_information_absent():
    """Test 4: When information is absent, assistant declares insufficient context without fabricating."""
    # Empty vector store
    empty_store = CourseVectorStore(in_memory=True)
    empty_store.clear()

    retriever = CourseRetriever(vector_store=empty_store)
    chunks, citations, is_grounded = retriever.retrieve("Quantum computing entanglement algorithms")

    assert is_grounded is False
    assert len(chunks) == 0
    assert len(citations) == 0

    state = AgentState(query="What does my course say about quantum entanglement algorithms?")
    state.retrieved_chunks = chunks
    state.sources = citations

    from app.agents.researcher import run_researcher_agent
    import app.agents.researcher as r_mod
    r_mod.course_retriever = retriever

    result = run_researcher_agent(state)

    # Must declare insufficient information
    assert "does not contain sufficient information" in result.final_response.lower()
    # Must NOT fabricate citations
    assert len(result.sources) == 0
