"""Document loader for PDFs and text files with page-level tracking."""

import os
from pathlib import Path
from typing import List, Dict, Any, Union
from pypdf import PdfReader


class DocumentPage:
    """Represents a single page or section from a document."""

    def __init__(self, content: str, source: str, page_number: int):
        self.content = content.strip()
        self.source = source
        self.page_number = page_number

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "page_number": self.page_number,
        }


class PDFDocumentLoader:
    """Loads PDF files and extracts text per page."""

    @staticmethod
    def load_pdf(file_path: Union[str, Path]) -> List[DocumentPage]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        pages: List[DocumentPage] = []
        source_name = path.name

        try:
            reader = PdfReader(str(path))
            for idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                # Normalize line breaks and spaces
                cleaned_text = " ".join(text.split())
                if cleaned_text:
                    pages.append(DocumentPage(cleaned_text, source_name, idx))
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF '{source_name}': {e}")

        return pages

    @staticmethod
    def load_text(file_path: Union[str, Path]) -> List[DocumentPage]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        cleaned_text = " ".join(text.split())
        return [DocumentPage(cleaned_text, path.name, 1)]
