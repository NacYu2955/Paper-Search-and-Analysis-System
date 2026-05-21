import logging
from typing import List

import numpy as np

from .fusion import reciprocal_rank_fusion
from .loader import PapersDBLoader
from .milvus_store import create_vector_store
from .sparse_store import BM25SparseStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Paper-level retrieval pipeline backed by Milvus and BM25.

    Data source: papers.db.
    Chunking strategy: one paper record is one retrieval chunk.
    Retrieval modes: dense, sparse, and hybrid RRF.
    """

    def __init__(
        self,
        *,
        embedding_model,
        database_path: str,
        milvus_uri: str,
        collection_name: str,
        enable_milvus: bool,
        enable_fallback: bool,
        max_text_chars: int,
        embed_batch_size: int,
        rrf_k: int = 60,
    ):
        self.embedding_model = embedding_model
        self.database_path = database_path
        self.max_text_chars = max_text_chars
        self.embed_batch_size = embed_batch_size
        self.rrf_k = rrf_k
        self.documents = []
        self.documents_by_id = {}
        self.embeddings = None
        self.sparse_store = BM25SparseStore()

        embedding_dim = self._get_embedding_dim()
        self.vector_store = create_vector_store(
            enable_milvus=enable_milvus,
            enable_fallback=enable_fallback,
            uri=milvus_uri,
            collection_name=collection_name,
            embedding_dim=embedding_dim,
        )

    @property
    def backend_name(self) -> str:
        return self.vector_store.backend_name

    def backend_name_for_mode(self, mode: str) -> str:
        if mode == "dense":
            return self.vector_store.backend_name
        if mode == "sparse":
            return self.sparse_store.backend_name
        if mode == "hybrid":
            return f"hybrid_rrf({self.vector_store.backend_name}+{self.sparse_store.backend_name})"
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    def _get_embedding_dim(self) -> int:
        if hasattr(self.embedding_model, "get_sentence_embedding_dimension"):
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            if dimension:
                return int(dimension)

        probe = self.embedding_model.encode(["dimension probe"])
        return int(np.asarray(probe)[0].shape[-1])

    def load_documents(self):
        loader = PapersDBLoader(
            database_path=self.database_path,
            max_text_chars=self.max_text_chars,
        )
        self.documents = loader.load_documents()
        self.documents_by_id = {doc.paper_id: doc for doc in self.documents}
        return self.documents

    def build_index(self, force: bool = False):
        documents = self.load_documents()
        if not documents:
            raise ValueError("papers.db数据库为空")

        self.sparse_store.build(documents)

        current_count = self.vector_store.count()
        should_embed = force or current_count != len(documents) or self.embeddings is None

        if should_embed:
            texts = [doc.chunk_text for doc in documents]
            logger.info("Embedding %s paper chunks for RAG retrieval", len(texts))
            self.embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                batch_size=self.embed_batch_size,
                convert_to_numpy=True,
            )
            self.vector_store.upsert_documents(documents, self.embeddings)
        else:
            logger.info(
                "Milvus collection already has %s rows; keeping existing vectors",
                current_count,
            )

        self.vector_store.load()
        return documents

    def retrieve(self, query: str, top_k: int, mode: str = "dense") -> List[dict]:
        if not self.documents_by_id:
            self.build_index(force=False)

        mode = mode.lower()
        if mode == "dense":
            hits = self._dense_hits(query, top_k=top_k)
        elif mode == "sparse":
            hits = self._sparse_hits(query, top_k=top_k)
        elif mode == "hybrid":
            dense_hits = self._dense_hits(query, top_k=top_k)
            sparse_hits = self._sparse_hits(query, top_k=top_k)
            hits = reciprocal_rank_fusion(
                {"dense": dense_hits, "sparse": sparse_hits},
                top_k=top_k,
                rrf_k=self.rrf_k,
            )
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        return self._format_hits(hits, backend_name=self.backend_name_for_mode(mode), mode=mode)

    def _dense_hits(self, query: str, top_k: int) -> List[dict]:
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        return self.vector_store.search(query_embedding, top_k=top_k)

    def _sparse_hits(self, query: str, top_k: int) -> List[dict]:
        return self.sparse_store.search(query, top_k=top_k)

    def _format_hits(self, hits: List[dict], backend_name: str, mode: str) -> List[dict]:
        results = []
        seen_ids = set()
        for rank, hit in enumerate(hits, start=1):
            paper_id = int(hit["paper_id"])
            if paper_id in seen_ids:
                continue
            doc = self.documents_by_id.get(paper_id)
            if doc is None:
                continue
            seen_ids.add(paper_id)
            branch_scores = hit.get("branch_scores", {})
            branch_ranks = hit.get("branch_ranks", {})
            dense_score = branch_scores.get("dense")
            sparse_score = branch_scores.get("sparse")
            dense_rank = branch_ranks.get("dense")
            sparse_rank = branch_ranks.get("sparse")
            if mode == "dense":
                dense_score = float(hit["score"])
                dense_rank = rank
            elif mode == "sparse":
                sparse_score = float(hit["score"])
                sparse_rank = rank
            results.append(
                {
                    "paper": doc.paper,
                    "similarity": float(hit["score"]),
                    "score": float(hit["score"]),
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "dense_rank": dense_rank,
                    "sparse_rank": sparse_rank,
                    "chunk_text": doc.chunk_text,
                    "retrieval_backend": backend_name,
                }
            )
        return results
