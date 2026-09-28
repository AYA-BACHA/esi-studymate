"""Test 2: Document retrieval and vector search pipeline."""

import pytest
from app.rag.loader import DocumentPage, PDFDocumentLoader
from app.rag.chunker import RecursiveChunker
from app.rag.vectorstore import CourseVectorStore
from app.rag.retriever import CourseRetriever


def test_chunking_preserves_metadata():
    """Verify that chunking preserves source and page numbers."""
    page = DocumentPage(
        content="Virtual memory allows an OS to compensate for physical memory shortages. Paging divides memory into fixed blocks.",
        source="Test_OS.pdf",
        page_number=14,
    )
    chunker = RecursiveChunker(chunk_size=60, chunk_overlap=15)
    chunks = chunker.chunk_page(page)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.source == "Test_OS.pdf"
        assert c.page_number == 14
        assert len(c.content) <= 80


def test_vectorstore_indexing_and_search():
    """Verify in-memory vector store indexing and top-k retrieval."""
    store = CourseVectorStore(in_memory=True)
    store.clear()

    page1 = DocumentPage(
        content="An interrupt is an asynchronous signal sent by hardware to the CPU to request service via an ISR.",
        source="Architecture_Lecture.pdf",
        page_number=3,
    )
    page2 = DocumentPage(
        content="Deadlock occurs when four conditions hold: mutual exclusion, hold and wait, no preemption, circular wait.",
        source="Operating_Systems.pdf",
        page_number=45,
    )

    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_documents([page1, page2])
    store.add_chunks(chunks)

    # Search for interrupts
    retriever = CourseRetriever(vector_store=store)
    results, citations, is_grounded = retriever.retrieve("interrupt signal to CPU", top_k=1)

    assert is_grounded is True
    assert len(results) == 1
    assert "interrupt" in results[0]["content"].lower()
    assert citations[0].source == "Architecture_Lecture.pdf"
    assert citations[0].page == 3
