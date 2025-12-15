from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

from docx import Document
from pypdf import PdfReader

ALLOWED_EXTENSIONS = {".txt", ".docx", ".pdf"}


def validate_extension(filename: str) -> bool:
    """Return True if filename extension is allowed."""
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """
    Convert uploaded file bytes into plain text.

    Supports .txt, .docx, .pdf.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return data.decode("utf-8", errors="ignore")
    if suffix == ".docx":
        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(data))
        buffer: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            extracted = extracted.strip()
            if extracted:
                buffer.append(extracted)
        return "\n".join(buffer)
    raise ValueError(f"Unsupported file type for {filename}")


def summarize_text(chunks: Iterable[str], max_chars: int = 4000) -> str:
    """
    Merge text chunks and trim to max_chars for safe prompting.
    """
    combined = "\n\n---\n\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip())
    if len(combined) <= max_chars:
        return combined
    return combined[:max_chars] + "\n\n...[truncated]..."

