"""PDF loading and chunking. Every chunk keeps its source file and page number."""
import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """Return a list of chunks: {text, source, page}."""
    chunks = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue  # scanned/empty page (no OCR in this version)
            for piece in _splitter.split_text(text):
                chunks.append({"text": piece, "source": filename, "page": page_num})
    return chunks