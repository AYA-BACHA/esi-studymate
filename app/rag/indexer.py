"""Document indexing pipeline coordinating loading, chunking, and Chroma persistence."""

from pathlib import Path
from typing import Union, Dict, Any, List
from app.rag.loader import PDFDocumentLoader, DocumentPage
from app.rag.chunker import RecursiveChunker, Chunk
from app.rag.vectorstore import course_vector_store, CourseVectorStore
from app.middleware.logging import observer


class DocumentIndexer:
    """Indexes PDF and text documents into the course vector store."""

    def __init__(self, vector_store: CourseVectorStore = course_vector_store):
        self.vector_store = vector_store
        self.chunker = RecursiveChunker(chunk_size=500, chunk_overlap=80)

    def index_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load, chunk, and index a file into ChromaDB."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        observer.record_event(
            event_type="agent_exec",
            name="Document Indexer",
            display_message=f"Indexing document: '{path.name}'...",
        )

        # 1. Extraction
        if path.suffix.lower() == ".pdf":
            pages = PDFDocumentLoader.load_pdf(path)
        else:
            pages = PDFDocumentLoader.load_text(path)

        if not pages:
            return {"file_name": path.name, "indexed_chunks": 0, "pages": 0}

        # 2. Chunking
        chunks = self.chunker.chunk_documents(pages)

        # 3. Vector Storage
        count = self.vector_store.add_chunks(chunks)

        observer.record_event(
            event_type="complete",
            name="Document Indexer",
            display_message=f"Indexed {count} chunks across {len(pages)} pages from '{path.name}'",
            metadata={"file": path.name, "chunks": count, "pages": len(pages)},
        )

        return {
            "file_name": path.name,
            "indexed_chunks": count,
            "pages": len(pages),
        }

    def index_bytes(self, content_bytes: bytes, file_name: str) -> Dict[str, Any]:
        """Index uploaded file bytes (from Streamlit uploader) by saving to data/documents/."""
        dest_dir = Path("./data/documents")
        dest_dir.mkdir(parents=True, exist_ok=True)
        target_path = dest_dir / file_name

        with open(target_path, "wb") as f:
            f.write(content_bytes)

        return self.index_file(target_path)


# Global indexer
document_indexer = DocumentIndexer()
