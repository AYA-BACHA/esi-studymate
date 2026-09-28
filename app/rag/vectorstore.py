"""Chroma vector database manager for course materials."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.rag.chunker import Chunk
from app.rag.embeddings import get_embedding_function


class CourseVectorStore:
    """Manages indexing and similarity search in ChromaDB."""

    COLLECTION_NAME = "esi_course_materials"

    def __init__(self, persist_directory: Optional[str] = None, in_memory: bool = False):
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.in_memory = in_memory
        self.embedding_fn = get_embedding_function()

        if self.in_memory:
            self.client = chromadb.EphemeralClient()
        else:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"description": "ESI StudyMate Course Materials"},
        )

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Add text chunks with page metadata to the vector collection."""
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {"source": chunk.source, "page": chunk.page_number}
            for chunk in chunks
        ]

        # Upsert into Chroma collection
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query vector store for the top_k most similar document chunks."""
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = self.collection.query(
            query_texts=[query],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                # Convert distance to similarity score
                similarity = max(0.0, 1.0 - float(dist)) if dist is not None else 1.0
                formatted_results.append({
                    "content": doc,
                    "source": meta.get("source", "Unknown Document"),
                    "page": meta.get("page", 1),
                    "score": round(similarity, 4),
                })

        return formatted_results

    def clear(self) -> None:
        """Clear all stored documents from the collection."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the vector store."""
        total_chunks = self.collection.count()
        return {
            "total_chunks": total_chunks,
            "collection_name": self.COLLECTION_NAME,
            "persist_directory": self.persist_directory if not self.in_memory else "in-memory",
        }


# Global vector store instance
course_vector_store = CourseVectorStore()
