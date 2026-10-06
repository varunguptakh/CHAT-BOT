"""Load the SecureGate 2FA PDF into LangChain Documents and chunk them for FAISS."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PDF_PATH = ROOT / "data" / "securegate_2fa_knowledge.pdf"
QA_PAIRS_PATH = ROOT / "data" / "qa_pairs.json"

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_pdf_chunks(pdf_path: Path | None = None) -> list[Document]:
    path = pdf_path or PDF_PATH
    if not path.exists():
        return []
    reader = PdfReader(str(path))
    pages: list[Document] = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        pages.append(
            Document(
                page_content=text,
                metadata={"source": f"{path.name}#page-{i}", "page": i, "kind": "pdf"},
            )
        )
    if not pages:
        return []
    return SPLITTER.split_documents(pages)
