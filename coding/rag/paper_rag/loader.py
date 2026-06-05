import sqlite3
from typing import List

from .documents import PaperDocument, paper_to_document


class PapersDBLoader:
    """Load approved paper records from the project SQLite database."""

    def __init__(self, database_path: str, max_text_chars: int):
        self.database_path = database_path
        self.max_text_chars = max_text_chars

    def load_documents(self) -> List[PaperDocument]:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM papers ORDER BY id").fetchall()
            return [
                paper_to_document(dict(row), max_chars=self.max_text_chars)
                for row in rows
            ]
        finally:
            conn.close()
