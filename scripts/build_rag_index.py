import argparse
import logging
import os
import sys

from sentence_transformers import SentenceTransformer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (  # noqa: E402
    DATABASE_PATH,
    EMBEDDING_MODEL_NAME,
    MODEL_PATH,
    RAG_EMBED_BATCH_SIZE,
    RAG_ENABLE_IN_MEMORY_FALLBACK,
    RAG_ENABLE_MILVUS,
    RAG_MAX_TEXT_CHARS,
    RAG_MILVUS_COLLECTION,
    RAG_MILVUS_URI,
    RAG_RRF_K,
)
from src.paper_rag import RAGPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build the Milvus RAG index from papers.db")
    parser.add_argument("--force", action="store_true", help="Re-embed and upsert all papers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    model_source = MODEL_PATH if os.path.exists(MODEL_PATH) else EMBEDDING_MODEL_NAME
    print(f"Embedding model: {model_source}")
    model = SentenceTransformer(model_source)
    pipeline = RAGPipeline(
        embedding_model=model,
        database_path=DATABASE_PATH,
        milvus_uri=RAG_MILVUS_URI,
        collection_name=RAG_MILVUS_COLLECTION,
        enable_milvus=RAG_ENABLE_MILVUS,
        enable_fallback=RAG_ENABLE_IN_MEMORY_FALLBACK,
        max_text_chars=RAG_MAX_TEXT_CHARS,
        embed_batch_size=RAG_EMBED_BATCH_SIZE,
        rrf_k=RAG_RRF_K,
    )
    documents = pipeline.build_index(force=args.force)
    print(
        f"Indexed {len(documents)} papers into {pipeline.backend_name}: "
        f"{RAG_MILVUS_COLLECTION}"
    )


if __name__ == "__main__":
    main()
