"""RAG (Retrieval-Augmented Generation) pipeline for ESI StudyMate.

Follows the simple LangChain pipeline:
1. Load PDF / Text document
2. Split text into chunks
3. Generate embeddings
4. Store in Chroma vector store
5. Retrieve relevant chunks for student questions
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb.utils.embedding_functions as ef

# Path for Chroma database
CHROMA_PATH = "./data/chroma_db"
COLLECTION_NAME = "course_material"

# Simple local embedding function (all-MiniLM-L6-v2)
embedding_function = ef.DefaultEmbeddingFunction()

# Global vector store instance
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_PATH,
)


def load_and_index_document(file_path: str) -> int:
    """Load a PDF or text file, split into chunks, and save to Chroma vector store."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load document
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    documents = loader.load()

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)

    # 3 & 4. Store chunks in vector store
    if chunks:
        vector_store.add_documents(chunks)

    return len(chunks)


def search_documents(query: str, top_k: int = 3) -> str:
    """Retrieve relevant chunks from the vector store and return as formatted text."""
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant course material found for this question."

    results = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "course document")).name
        page = doc.metadata.get("page", 1)
        results.append(f"--- Source: {source} (Page {page}) ---\n{doc.page_content.strip()}")

    return "\n\n".join(results)
