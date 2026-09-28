"""Text chunking with page and source metadata preservation."""

from typing import List, Dict, Any
from app.rag.loader import DocumentPage


class Chunk:
    """A semantic chunk of text with document citation metadata."""

    def __init__(self, content: str, source: str, page_number: int, chunk_id: str):
        self.content = content
        self.source = source
        self.page_number = page_number
        self.chunk_id = chunk_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "page_number": self.page_number,
        }


class RecursiveChunker:
    """Splits document pages into overlapping text chunks."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page(self, page: DocumentPage) -> List[Chunk]:
        """Split a single DocumentPage into overlapping chunks."""
        text = page.content
        if not text:
            return []

        chunks: List[Chunk] = []
        start = 0
        text_len = len(text)
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Try to snap to sentence or whitespace boundary if not at end of text
            if end < text_len:
                last_punct = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("\n", start, end),
                )
                if last_punct != -1 and last_punct > start + (self.chunk_size // 2):
                    end = last_punct + 1
                else:
                    last_space = text.rfind(" ", start, end)
                    if last_space != -1 and last_space > start + (self.chunk_size // 2):
                        end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"{page.source}_p{page.page_number}_c{chunk_idx}"
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        source=page.source,
                        page_number=page.page_number,
                        chunk_id=chunk_id,
                    )
                )
                chunk_idx += 1

            if end >= text_len:
                break

            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def chunk_documents(self, pages: List[DocumentPage]) -> List[Chunk]:
        """Process a list of pages into all chunks."""
        all_chunks: List[Chunk] = []
        for page in pages:
            all_chunks.extend(self.chunk_page(page))
        return all_chunks
