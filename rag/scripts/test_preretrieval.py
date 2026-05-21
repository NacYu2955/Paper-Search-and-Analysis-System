import argparse
import os
import sys

import torch
from sentence_transformers import SentenceTransformer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.config import (  # noqa: E402
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
from rag.paper_rag import RAGPipeline  # noqa: E402


def load_embedding_model():
    source = MODEL_PATH if os.path.exists(MODEL_PATH) else EMBEDDING_MODEL_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding model: {source}")
    print(f"Device: {device}")
    return SentenceTransformer(source, device=device)


def main():
    parser = argparse.ArgumentParser(description="Test Milvus dense pre-retrieval only")
    parser.add_argument("query", help="Query text for dense retrieval")
    parser.add_argument("--top-k", type=int, default=10, help="Number of papers to retrieve")
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="dense")
    parser.add_argument("--force", action="store_true", help="Rebuild vectors before searching")
    args = parser.parse_args()

    model = load_embedding_model()
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
    results = pipeline.retrieve(args.query, top_k=args.top_k, mode=args.mode)

    print(f"\nLoaded papers: {len(documents)}")
    print(f"Retrieval backend: {pipeline.backend_name_for_mode(args.mode)}")
    print(f"Query: {args.query}\n")
    for index, result in enumerate(results, 1):
        paper = result["paper"]
        print(
            f"{index}. [{result['similarity']:.4f}] "
            f"{paper.get('title', 'Untitled')} "
            f"({paper.get('year', 'Unknown')})"
        )


if __name__ == "__main__":
    main()
