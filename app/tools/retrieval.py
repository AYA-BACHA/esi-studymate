"""Course retrieval tool definition."""

from typing import Dict, Any, List
from app.rag.retriever import course_retriever


def search_course_materials(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Search uploaded university course slides, books, and notes.
    
    Args:
        query: Specific concept or keywords to look up in course documents.
        top_k: Number of most relevant document sections to retrieve.
    """
    chunks, citations, is_grounded = course_retriever.retrieve(query=query, top_k=top_k)
    return {
        "query": query,
        "found_count": len(chunks),
        "is_grounded": is_grounded,
        "chunks": chunks,
        "sources": [c.model_dump() for c in citations],
    }
