import math
import re
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, List, Sequence, Tuple

from .documents import PaperDocument


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*|[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")]


class BM25SparseStore:
    """Small in-memory BM25 index for the closed 446-paper corpus."""

    backend_name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[PaperDocument] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len = 0.0
        self.idf: Dict[str, float] = {}
        self.inverted_index: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)

    def build(self, documents: Sequence[PaperDocument]) -> None:
        self.documents = list(documents)
        self.doc_lens = []
        self.idf = {}
        self.inverted_index = defaultdict(list)

        doc_freqs: Counter[str] = Counter()
        term_counts_by_doc = []
        for doc in self.documents:
            tokens = tokenize(doc.chunk_text)
            term_counts = Counter(tokens)
            term_counts_by_doc.append(term_counts)
            self.doc_lens.append(len(tokens))
            for term in term_counts:
                doc_freqs[term] += 1

        total_docs = len(self.documents)
        self.avg_doc_len = sum(self.doc_lens) / total_docs if total_docs else 0.0
        for term, doc_freq in doc_freqs.items():
            self.idf[term] = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        for doc_index, term_counts in enumerate(term_counts_by_doc):
            for term, term_freq in term_counts.items():
                self.inverted_index[term].append((doc_index, term_freq))

    def count(self) -> int:
        return len(self.documents)

    def search(self, query: str, top_k: int) -> List[dict]:
        if not self.documents:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: DefaultDict[int, float] = defaultdict(float)
        for term in query_terms:
            postings = self.inverted_index.get(term)
            if not postings:
                continue

            idf = self.idf.get(term, 0.0)
            for doc_index, term_freq in postings:
                doc_len = self.doc_lens[doc_index] or 1
                norm = 1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0))
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * norm
                scores[doc_index] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {
                "paper_id": self.documents[doc_index].paper_id,
                "score": float(score),
            }
            for doc_index, score in ranked
        ]
