"""
Document loaders. Start with text files; PDF support added below.
"""
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def load_text_file(path: str) -> Dict:
    """Load a single text file into a document dict."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    return {"text": text, "source": p.name}


def load_directory(directory: str, pattern: str = "*.txt") -> List[Dict]:
    """Load all matching files from a directory."""
    docs = []
    for path in Path(directory).glob(pattern):
        try:
            docs.append(load_text_file(str(path)))
            logger.info("Loaded: %s", path.name)
        except Exception as e:
            logger.error("Failed to load %s: %s", path.name, e)
    logger.info("Loaded %d documents from %s", len(docs), directory)
    return docs


def load_pdf(path: str) -> Dict:
    """
    Load a PDF, extracting text from all pages.
    Requires: pip install pypdf
    """
    from pypdf import PdfReader

    p = Path(path)
    reader = PdfReader(str(p))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    full_text = "\n\n".join(text_parts)
    return {"text": full_text, "source": p.name, "num_pages": len(reader.pages)}