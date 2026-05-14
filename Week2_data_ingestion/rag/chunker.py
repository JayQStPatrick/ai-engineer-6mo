from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

def chunk_recursive(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return _apply_splitter(splitter, pages)

def chunk_token(
    pages: list[dict],
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> list[dict]:
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return _apply_splitter(splitter, pages)

def _apply_splitter(splitter, pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        texts = splitter.split_text(page["text"])
        for i, text in enumerate(texts):
            chunks.append({
                "text": text,
                "metadata": {
                    **page["metadata"],
                    "chunk": i,
                    "section": _detect_section(text),
                }
            })
    return chunks

def _detect_section(text: str) -> str:
    lines = text.strip().split("\n")
    return lines[0][:80] if lines else ""