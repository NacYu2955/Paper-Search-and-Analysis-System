from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PaperDocument:
    """A single retrievable unit for RAG.

    The current retrieval design keeps one complete paper as one chunk.
    """

    paper_id: int
    title: str
    authors: str
    year: Optional[int]
    venue: str
    chunk_text: str
    paper: Dict[str, Any]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def build_paper_chunk(paper: Dict[str, Any], max_chars: int) -> str:
    """Build one dense-retrieval chunk from one paper record."""

    title = normalize_text(paper.get("title"))
    authors = normalize_text(paper.get("authors") or paper.get("author"))
    year = normalize_text(paper.get("year"))
    venue = normalize_text(paper.get("journal") or paper.get("booktitle"))
    abstract = normalize_text(paper.get("abstract"))
    keywords = normalize_text(paper.get("keywords"))

    parts = [
        f"Title: {title}" if title else "",
        f"Authors: {authors}" if authors else "",
        f"Year: {year}" if year else "",
        f"Venue: {venue}" if venue else "",
        f"Keywords: {keywords}" if keywords else "",
        f"Abstract: {abstract}" if abstract else "",
    ]
    text = "\n".join(part for part in parts if part)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0]
    return text


def paper_to_document(paper: Dict[str, Any], max_chars: int) -> PaperDocument:
    paper_id = int(paper["id"])
    venue = normalize_text(paper.get("journal") or paper.get("booktitle"))
    year = paper.get("year")
    return PaperDocument(
        paper_id=paper_id,
        title=normalize_text(paper.get("title")),
        authors=normalize_text(paper.get("authors") or paper.get("author")),
        year=int(year) if str(year or "").isdigit() else None,
        venue=venue,
        chunk_text=build_paper_chunk(paper, max_chars=max_chars),
        paper=paper,
    )
