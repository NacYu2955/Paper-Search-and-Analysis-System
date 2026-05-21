import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Set

import torch
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
    SELECTOR_PATH,
)
from src.paper_rag import QwenReranker, RAGPipeline  # noqa: E402
from src.paper_rag.evaluation import (  # noqa: E402
    build_query_detail,
    compute_metrics,
    format_metric_line,
    load_jsonl,
    paper_ids,
    parse_eval_ks,
    write_details_csv,
    write_json,
    write_metrics_csv,
)


def load_embedding_model():
    source = MODEL_PATH if os.path.exists(MODEL_PATH) else EMBEDDING_MODEL_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding model: {source}")
    print(f"Embedding device: {device}")
    return SentenceTransformer(source, device=device)


def apply_rerank(
    candidates: List[dict],
    query: str,
    reranker: QwenReranker,
    score_threshold: Optional[float],
    final_k: int,
) -> List[dict]:
    scores = reranker.score(candidates, query)
    for result, score in zip(candidates, scores):
        result["select_score"] = float(score)

    ranked = sorted(
        candidates,
        key=lambda item: (item.get("select_score", 0.0), item.get("similarity", 0.0)),
        reverse=True,
    )

    seen_titles: Set[str] = set()
    unique_ranked = []
    for result in ranked:
        title = result["paper"].get("title", "").lower().strip()
        if title in seen_titles:
            continue
        seen_titles.add(title)
        unique_ranked.append(result)

    if score_threshold is None:
        return unique_ranked[:final_k]

    filtered = [
        result
        for result in unique_ranked
        if result.get("select_score", 0.0) > score_threshold
    ]
    if not filtered:
        return unique_ranked[:1]
    return filtered[:final_k]


def percentile(values: List[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate closed-corpus retrieval/rerank on a JSONL ground-truth set"
    )
    parser.add_argument("--testset", default="deepseek_testset(1).jsonl")
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "sparse", "hybrid", "all"],
        default="dense",
        help="Retrieval branch to evaluate before optional rerank.",
    )
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--eval-ks", default="1,3,5,10,20,50")
    parser.add_argument("--output-dir", default=os.path.join("output", "eval"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--rerank", action="store_true", help="Run PASA/Qwen rerank after dense retrieval")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Rerank score threshold. Use --no-threshold to disable threshold filtering.",
    )
    parser.add_argument("--no-threshold", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(args.testset)
    requested_ks = parse_eval_ks(args.eval_ks)
    candidate_eval_ks = sorted({k for k in requested_ks + [args.candidate_k] if k <= args.candidate_k})
    final_eval_ks = [k for k in requested_ks if k <= args.final_k]

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
    documents = pipeline.build_index(force=args.force_index)

    reranker = None
    threshold = None if args.no_threshold else args.score_threshold
    if args.rerank:
        if not os.path.exists(SELECTOR_PATH):
            raise FileNotFoundError(f"Rerank model path does not exist: {SELECTOR_PATH}")
        reranker = QwenReranker(
            SELECTOR_PATH,
            prompt_path=os.path.join(ROOT_DIR, "agent_prompt.json"),
        )

    os.makedirs(args.output_dir, exist_ok=True)
    run_modes = ["dense", "sparse", "hybrid"] if args.retrieval_mode == "all" else [args.retrieval_mode]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_summaries = []

    for retrieval_mode in run_modes:
        candidate_predictions = {}
        final_predictions = {}
        details = []
        latencies = []

        for record in records:
            start = time.perf_counter()
            candidates = pipeline.retrieve(
                record["question"],
                top_k=args.candidate_k,
                mode=retrieval_mode,
            )
            candidate_predictions[record["qid"]] = paper_ids(candidates)

            if reranker is not None:
                final_results = apply_rerank(
                    candidates=candidates,
                    query=record["question"],
                    reranker=reranker,
                    score_threshold=threshold,
                    final_k=args.final_k,
                )
            else:
                final_results = candidates[:args.final_k]

            final_predictions[record["qid"]] = paper_ids(final_results)
            latency = time.perf_counter() - start
            latencies.append(latency)
            details.append(
                build_query_detail(
                    record=record,
                    candidate_results=candidates,
                    final_results=final_results,
                    candidate_k=args.candidate_k,
                    final_k=args.final_k,
                    latency_seconds=latency,
                )
            )

        print(f"\n=== Retrieval mode: {retrieval_mode} ===")
        print(f"Testset: {args.testset}")
        print(f"Queries: {len(records)}")
        print(f"Corpus papers: {len(documents)}")
        print(f"Retrieval backend: {pipeline.backend_name_for_mode(retrieval_mode)}")
        print(f"Candidate K: {args.candidate_k}")
        print(f"Rerank: {bool(reranker)}")
        print(f"Final K: {args.final_k}")
        if reranker is not None:
            print(f"Score threshold: {'disabled' if threshold is None else threshold}")
        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p50_latency = percentile(latencies, 0.50)
        p95_latency = percentile(latencies, 0.95)
        print(f"Mean latency/query: {mean_latency:.3f}s")
        print(f"P50 latency/query: {p50_latency:.3f}s")
        print(f"P95 latency/query: {p95_latency:.3f}s")

        candidate_metrics = compute_metrics(records, candidate_predictions, candidate_eval_ks)
        print("\nCandidate recall stage")
        for k in candidate_eval_ks:
            print(format_metric_line("Candidate", k, candidate_metrics[k]))

        final_metrics = compute_metrics(records, final_predictions, final_eval_ks)
        print("\nFinal system output")
        for k in final_eval_ks:
            print(format_metric_line("Final", k, final_metrics[k]))

        if args.run_name is None:
            stage = "rerank" if reranker is not None else "base"
            run_name = f"{timestamp}_{retrieval_mode}_{stage}_c{args.candidate_k}_f{args.final_k}"
        elif len(run_modes) > 1:
            run_name = f"{args.run_name}_{retrieval_mode}"
        else:
            run_name = args.run_name

        summary_path = os.path.join(args.output_dir, f"{run_name}_summary.json")
        details_path = os.path.join(args.output_dir, f"{run_name}_details.json")
        details_csv_path = os.path.join(args.output_dir, f"{run_name}_details.csv")
        metrics_csv_path = os.path.join(args.output_dir, f"{run_name}_metrics.csv")

        summary = {
            "run_name": run_name,
            "testset": args.testset,
            "queries": len(records),
            "corpus_papers": len(documents),
            "retrieval_mode": retrieval_mode,
            "retrieval_backend": pipeline.backend_name_for_mode(retrieval_mode),
            "embedding_model": MODEL_PATH if os.path.exists(MODEL_PATH) else EMBEDDING_MODEL_NAME,
            "candidate_k": args.candidate_k,
            "final_k": args.final_k,
            "rrf_k": RAG_RRF_K if retrieval_mode == "hybrid" else None,
            "rerank": bool(reranker),
            "score_threshold": None if reranker is None or threshold is None else threshold,
            "mean_latency_seconds": mean_latency,
            "p50_latency_seconds": p50_latency,
            "p95_latency_seconds": p95_latency,
            "candidate_metrics": candidate_metrics,
            "final_metrics": final_metrics,
        }
        all_summaries.append(summary)
        write_json(summary_path, summary)
        write_json(details_path, {"summary": summary, "details": details})
        write_details_csv(details_csv_path, details)
        write_metrics_csv(metrics_csv_path, [summary])

        print("\nSaved evaluation artifacts")
        print(f"Summary: {summary_path}")
        print(f"Details JSON: {details_path}")
        print(f"Details CSV: {details_csv_path}")
        print(f"Metrics CSV: {metrics_csv_path}")

    if len(all_summaries) > 1:
        aggregate_name = args.run_name or f"{timestamp}_all_c{args.candidate_k}_f{args.final_k}"
        aggregate_csv_path = os.path.join(args.output_dir, f"{aggregate_name}_metrics.csv")
        write_metrics_csv(aggregate_csv_path, all_summaries)
        print(f"\nAggregate metrics CSV: {aggregate_csv_path}")


if __name__ == "__main__":
    main()
