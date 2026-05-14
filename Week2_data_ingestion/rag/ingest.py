from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
import markdown

def load_document(path: str) -> list[dict]:
    """Load a file and return a list of pages/sections with metadata."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(p)
    elif suffix in (".html", ".htm"):
        return _load_html(p)
    elif suffix in (".md", ".markdown"):
        return _load_markdown(p)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def _load_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({
            "text": text.strip(),
            "metadata": {
                "source": str(path),
                "page": i + 1,
                "type": "pdf"
            }
        })
    return pages

def _load_html(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [{"text": text, "metadata": {"source": str(path),
                                         "page": 1, "type": "html"}}]

def _load_markdown(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    html = markdown.markdown(raw)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return [{"text": text, "metadata": {"source": str(path),
                                         "page": 1, "type": "markdown"}}]