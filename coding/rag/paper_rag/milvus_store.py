import logging
import os
from typing import List, Optional, Sequence

import numpy as np

from .documents import PaperDocument

logger = logging.getLogger(__name__)

try:
    from pymilvus import DataType, MilvusClient
except ImportError:  # pragma: no cover - optional runtime dependency
    DataType = None
    MilvusClient = None


class MilvusUnavailableError(RuntimeError):
    pass


class MilvusVectorStore:
    backend_name = "milvus"

    def __init__(
        self,
        uri: str,
        collection_name: str,
        embedding_dim: int,
        metric_type: str = "COSINE",
    ):
        if MilvusClient is None:
            raise MilvusUnavailableError("pymilvus is not installed")

        self.uri = uri
        self.collection_name = collection_name
        self.embedding_dim = int(embedding_dim)
        self.metric_type = metric_type
        self.vector_field = "embedding"
        self.primary_field = "paper_id"

        self._ensure_local_parent_dir(uri)
        self.client = MilvusClient(uri=uri)
        self._ensure_collection()

    def _ensure_local_parent_dir(self, uri: str) -> None:
        if uri and "://" not in uri:
            parent = os.path.dirname(os.path.abspath(uri))
            if parent:
                os.makedirs(parent, exist_ok=True)

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            return

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.embedding_dim,
                primary_field_name=self.primary_field,
                vector_field_name=self.vector_field,
                metric_type=self.metric_type,
                auto_id=False,
            )
            return
        except TypeError:
            logger.info("Falling back to explicit Milvus schema creation")

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(self.primary_field, DataType.INT64, is_primary=True)
        schema.add_field(self.vector_field, DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("authors", DataType.VARCHAR, max_length=2048)
        schema.add_field("year", DataType.INT64)
        schema.add_field("venue", DataType.VARCHAR, max_length=1024)
        schema.add_field("chunk_text", DataType.VARCHAR, max_length=8192)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=self.vector_field,
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def count(self) -> int:
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def upsert_documents(
        self,
        documents: Sequence[PaperDocument],
        embeddings: np.ndarray,
        batch_size: int = 128,
    ) -> None:
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            rows = []
            for doc, embedding in zip(batch_docs, batch_embeddings):
                rows.append(
                    {
                        self.primary_field: doc.paper_id,
                        self.vector_field: np.asarray(embedding, dtype=np.float32).tolist(),
                        "title": doc.title[:1024],
                        "authors": doc.authors[:2048],
                        "year": int(doc.year or 0),
                        "venue": doc.venue[:1024],
                        "chunk_text": doc.chunk_text[:8192],
                    }
                )
            if rows:
                self.client.upsert(collection_name=self.collection_name, data=rows)

        try:
            self.client.flush(collection_name=self.collection_name)
        except Exception:
            pass
        self.load()

    def load(self) -> None:
        try:
            self.client.load_collection(self.collection_name)
        except Exception:
            pass

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[dict]:
        self.load()
        raw_results = self.client.search(
            collection_name=self.collection_name,
            data=[np.asarray(query_embedding, dtype=np.float32).tolist()],
            anns_field=self.vector_field,
            limit=top_k,
            output_fields=[self.primary_field, "title", "authors", "year", "venue"],
            search_params={"metric_type": self.metric_type},
        )
        hits = raw_results[0] if raw_results else []
        return [self._normalize_hit(hit) for hit in hits]

    def _normalize_hit(self, hit) -> dict:
        if isinstance(hit, dict):
            entity = hit.get("entity") or {}
            paper_id = entity.get(self.primary_field, hit.get("id"))
            score = hit.get("distance", hit.get("score", 0.0))
            return {"paper_id": int(paper_id), "score": float(score)}

        entity = getattr(hit, "entity", {}) or {}
        paper_id = entity.get(self.primary_field, getattr(hit, "id", None))
        score = getattr(hit, "distance", getattr(hit, "score", 0.0))
        return {"paper_id": int(paper_id), "score": float(score)}


class InMemoryVectorStore:
    backend_name = "in_memory"

    def __init__(self):
        self.documents: List[PaperDocument] = []
        self.embeddings: Optional[np.ndarray] = None

    def count(self) -> int:
        return len(self.documents)

    def upsert_documents(
        self,
        documents: Sequence[PaperDocument],
        embeddings: np.ndarray,
        batch_size: int = 128,
    ) -> None:
        self.documents = list(documents)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def load(self) -> None:
        return None

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[dict]:
        if self.embeddings is None or len(self.documents) == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query)
        denom = np.maximum(doc_norms * query_norm, 1e-12)
        scores = (self.embeddings @ query) / denom
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "paper_id": self.documents[int(index)].paper_id,
                "score": float(scores[int(index)]),
            }
            for index in indices
        ]


def create_vector_store(
    *,
    enable_milvus: bool,
    enable_fallback: bool,
    uri: str,
    collection_name: str,
    embedding_dim: int,
):
    if enable_milvus:
        try:
            return MilvusVectorStore(
                uri=uri,
                collection_name=collection_name,
                embedding_dim=embedding_dim,
            )
        except Exception as exc:
            if not enable_fallback:
                raise
            logger.warning("Milvus is unavailable, using in-memory retrieval: %s", exc)

    return InMemoryVectorStore()
