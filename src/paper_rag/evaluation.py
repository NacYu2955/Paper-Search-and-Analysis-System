import csv
import json
from typing import Dict, List, Optional, Sequence, Set


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def paper_ids(results: Sequence[dict]) -> List[int]:
    return [int(result["paper"]["id"]) for result in results]


def compact_results(results: Sequence[dict], limit: int) -> List[dict]:
    compacted = []
    for result in results[:limit]:
        paper = result["paper"]
        compacted.append(
            {
                "id": int(paper["id"]),
                "title": paper.get("title", ""),
                "similarity": float(result.get("similarity", 0.0)),
                "dense_score": result.get("dense_score"),
                "sparse_score": result.get("sparse_score"),
                "dense_rank": result.get("dense_rank"),
                "sparse_rank": result.get("sparse_rank"),
                "select_score": (
                    float(result["select_score"])
                    if "select_score" in result and result["select_score"] is not None
                    else None
                ),
            }
        )
    return compacted


def parse_eval_ks(value: str) -> List[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def matched_ids(ranked_ids: Sequence[int], gt_ids: Set[int], k: int) -> Set[int]:
    return set(ranked_ids[:k]) & gt_ids


def first_hit_rank(ranked_ids: Sequence[int], gt_ids: Set[int], k: int) -> Optional[int]:
    for index, paper_id in enumerate(ranked_ids[:k], start=1):
        if paper_id in gt_ids:
            return index
    return None


def compute_metrics(records: Sequence[dict], predictions: Dict[str, List[int]], ks: Sequence[int]):
    metrics = {}
    n = len(records)
    for k in ks:
        hit = 0
        recall_sum = 0.0
        all_gt = 0
        mrr_sum = 0.0

        for record in records:
            qid = record["qid"]
            gt_ids = set(int(pid) for pid in record.get("answer_paper_id", []))
            ranked_ids = predictions.get(qid, [])
            matched = matched_ids(ranked_ids, gt_ids, k)

            rank = first_hit_rank(ranked_ids, gt_ids, k)
            if rank is not None:
                hit += 1
                mrr_sum += 1.0 / rank

            if gt_ids:
                recall_sum += len(matched) / len(gt_ids)
                if matched == gt_ids:
                    all_gt += 1

        metrics[k] = {
            "hit": hit / n if n else 0.0,
            "recall": recall_sum / n if n else 0.0,
            "all_gt": all_gt / n if n else 0.0,
            "mrr": mrr_sum / n if n else 0.0,
        }
    return metrics


def build_query_detail(
    record: dict,
    candidate_results: Sequence[dict],
    final_results: Sequence[dict],
    candidate_k: int,
    final_k: int,
    latency_seconds: float,
) -> dict:
    gt_ids = set(int(pid) for pid in record.get("answer_paper_id", []))
    candidate_ranked_ids = paper_ids(candidate_results)
    final_ranked_ids = paper_ids(final_results)
    candidate_matched = matched_ids(candidate_ranked_ids, gt_ids, candidate_k)
    final_matched = matched_ids(final_ranked_ids, gt_ids, final_k)

    return {
        "qid": record.get("qid"),
        "question": record.get("question"),
        "answer_paper_id": list(record.get("answer_paper_id", [])),
        "answer": list(record.get("answer", [])),
        "candidate_hit": bool(candidate_matched),
        "candidate_recall": len(candidate_matched) / len(gt_ids) if gt_ids else 0.0,
        "candidate_first_hit_rank": first_hit_rank(candidate_ranked_ids, gt_ids, candidate_k),
        "candidate_matched_ids": sorted(candidate_matched),
        "final_hit": bool(final_matched),
        "final_recall": len(final_matched) / len(gt_ids) if gt_ids else 0.0,
        "final_first_hit_rank": first_hit_rank(final_ranked_ids, gt_ids, final_k),
        "final_matched_ids": sorted(final_matched),
        "latency_seconds": latency_seconds,
        "candidate_results": compact_results(candidate_results, candidate_k),
        "final_results": compact_results(final_results, final_k),
    }


def format_metric_line(prefix: str, k: int, metric: dict) -> str:
    return (
        f"{prefix}@{k:<3} "
        f"Hit={metric['hit']:.3f} "
        f"Recall={metric['recall']:.3f} "
        f"AllGT={metric['all_gt']:.3f} "
        f"MRR={metric['mrr']:.3f}"
    )


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_details_csv(path: str, details: Sequence[dict]) -> None:
    fieldnames = [
        "qid",
        "candidate_hit",
        "candidate_recall",
        "candidate_first_hit_rank",
        "final_hit",
        "final_recall",
        "final_first_hit_rank",
        "latency_seconds",
        "answer_paper_id",
        "candidate_matched_ids",
        "final_matched_ids",
        "question",
    ]
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for detail in details:
            writer.writerow(
                {
                    "qid": detail["qid"],
                    "candidate_hit": detail["candidate_hit"],
                    "candidate_recall": detail["candidate_recall"],
                    "candidate_first_hit_rank": detail["candidate_first_hit_rank"],
                    "final_hit": detail["final_hit"],
                    "final_recall": detail["final_recall"],
                    "final_first_hit_rank": detail["final_first_hit_rank"],
                    "latency_seconds": detail["latency_seconds"],
                    "answer_paper_id": " ".join(str(item) for item in detail["answer_paper_id"]),
                    "candidate_matched_ids": " ".join(str(item) for item in detail["candidate_matched_ids"]),
                    "final_matched_ids": " ".join(str(item) for item in detail["final_matched_ids"]),
                    "question": detail["question"],
                }
            )


def _metric_rows(summary: dict) -> List[dict]:
    rows = []
    common = {
        "run_name": summary.get("run_name"),
        "testset": summary.get("testset"),
        "queries": summary.get("queries"),
        "corpus_papers": summary.get("corpus_papers"),
        "retrieval_mode": summary.get("retrieval_mode"),
        "retrieval_backend": summary.get("retrieval_backend"),
        "embedding_model": summary.get("embedding_model"),
        "candidate_k": summary.get("candidate_k"),
        "final_k": summary.get("final_k"),
        "rrf_k": summary.get("rrf_k"),
        "rerank": summary.get("rerank"),
        "score_threshold": summary.get("score_threshold"),
        "mean_latency_seconds": summary.get("mean_latency_seconds"),
        "p50_latency_seconds": summary.get("p50_latency_seconds"),
        "p95_latency_seconds": summary.get("p95_latency_seconds"),
    }
    for stage_name, metric_group in [
        ("candidate", summary.get("candidate_metrics", {})),
        ("final", summary.get("final_metrics", {})),
    ]:
        for k, metric in metric_group.items():
            rows.append(
                {
                    **common,
                    "stage": stage_name,
                    "k": int(k),
                    "hit": metric.get("hit"),
                    "recall": metric.get("recall"),
                    "all_gt": metric.get("all_gt"),
                    "mrr": metric.get("mrr"),
                }
            )
    return rows


def write_metrics_csv(path: str, summaries: Sequence[dict]) -> None:
    fieldnames = [
        "run_name",
        "testset",
        "queries",
        "corpus_papers",
        "retrieval_mode",
        "retrieval_backend",
        "embedding_model",
        "candidate_k",
        "final_k",
        "rrf_k",
        "rerank",
        "score_threshold",
        "stage",
        "k",
        "hit",
        "recall",
        "all_gt",
        "mrr",
        "mean_latency_seconds",
        "p50_latency_seconds",
        "p95_latency_seconds",
    ]
    rows = []
    for summary in summaries:
        rows.extend(_metric_rows(summary))

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
