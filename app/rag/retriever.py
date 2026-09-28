"""Retriever interface for course materials with strict grounding checks."""

from typing import List, Dict, Any, Tuple
from app.config import settings
from app.state import SourceCitation
from app.rag.vectorstore import course_vector_store, CourseVectorStore


class CourseRetriever:
    """Searches course materials and formats sources with grounding validation."""

    def __init__(self, vector_store: CourseVectorStore = course_vector_store):
        self.vector_store = vector_store

    def retrieve(
        self, query: str, top_k: int = 3, min_score: float = 0.15
    ) -> Tuple[List[Dict[str, Any]], List[SourceCitation], bool]:
        """Retrieve relevant chunks and determine if grounding is sufficient.
        
        Returns:
            (retrieved_chunks, citations, is_grounded)
        """
        raw_results = self.vector_store.similarity_search(query, top_k=top_k)

        if not raw_results:
            return [], [], False

        citations: List[SourceCitation] = []
        filtered_chunks: List[Dict[str, Any]] = []

        for item in raw_results:
            preview = item["content"][:120] + "..." if len(item["content"]) > 120 else item["content"]
            citation = SourceCitation(
                source=item["source"],
                page=item["page"],
                preview=preview,
                score=item.get("score", 0.0),
            )
            citations.append(citation)
            filtered_chunks.append(item)

        # Grounding condition: at least one relevant chunk
        is_grounded = len(filtered_chunks) > 0

        return filtered_chunks, citations, is_grounded

    def format_context_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks safely into prompt context XML tags."""
        if not chunks:
            return "No relevant course documents found in the database."

        formatted_blocks = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "Document")
            page = chunk.get("page", 1)
            content = chunk.get("content", "")
            formatted_blocks.append(
                f'<course_document index="{i}" source="{source}" page="{page}">\n{content}\n</course_document>'
            )

        return "\n\n".join(formatted_blocks)


# Global retriever
course_retriever = CourseRetriever()
