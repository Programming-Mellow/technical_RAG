"""
Ingestion pipeline: load PDFs → chunk → embed → store in ChromaDB.

Usage:
    python ingest.py --corpus-dir ./docs
"""

import argparse
from pathlib import Path

import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Targets the 300–500 token range (tiktoken cl100k: ~4 chars/token).
# chunk_size=1600 ≈ 400 tokens; chunk_overlap=200 ≈ 50 tokens.
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200


def load_pdfs(corpus_dir: str) -> list:
    docs = []
    pdf_files = list(Path(corpus_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {corpus_dir}")

    for pdf_path in pdf_files:
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages):
                docs.append(Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": str(pdf_path), "page": i},
                ))
            print(f"  Loaded {len(reader.pages):>4} pages  ←  {pdf_path.name}")
        except Exception as exc:
            print(f"  WARNING: skipping {pdf_path.name} — {exc}")

    return docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest(corpus_dir: str) -> None:
    print(f"\n[1/3] Loading PDFs from '{corpus_dir}' ...")
    docs = load_pdfs(corpus_dir)
    unique_sources = len({d.metadata["source"] for d in docs})
    print(f"      {len(docs)} pages loaded from {unique_sources} documents")

    print(f"\n[2/3] Chunking (size={CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP} chars) ...")
    chunks = chunk_documents(docs)
    print(f"      {len(chunks)} chunks created")

    print(f"\n[3/3] Embedding with '{EMBED_MODEL}' and writing to ChromaDB ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"\nDone. {len(chunks)} chunks stored in '{CHROMA_DIR}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument(
        "--corpus-dir",
        default="./docs",
        help="Directory containing PDF files (default: ./docs)",
    )
    args = parser.parse_args()
    ingest(args.corpus_dir)
