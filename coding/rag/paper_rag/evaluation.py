import csv
import json
import math
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


def ndcg_at_k(ranked_ids: Sequence[int], gt_ids: Set[int], k: int) -> float:
    if not gt_ids:
        return 0.0

    dcg = 0.0
    for index, paper_id in enumerate(ranked_ids[:k], start=1):
        if paper_id in gt_ids:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(gt_ids), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def compute_metrics(records: Sequence[dict], predictions: Dict[str, List[int]], ks: Sequence[int]):
    metrics = {}
    n = len(records)
    for k in ks:
        hit = 0
        recall_sum = 0.0
        all_gt = 0
        mrr_sum = 0.0
        ndcg_sum = 0.0

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
                ndcg_sum += ndcg_at_k(ranked_ids, gt_ids, k)
                if matched == gt_ids:
                    all_gt += 1

        metrics[k] = {
            "hit": hit / n if n else 0.0,
            "recall": recall_sum / n if n else 0.0,
            "all_gt": all_gt / n if n else 0.0,
            "mrr": mrr_sum / n if n else 0.0,
            "ndcg": ndcg_sum / n if n else 0.0,
        }
    return metrics


def build_report_metrics(
    metrics: Dict[int, dict],
    latency_seconds: float,
    *,
    recall_ks: Sequence[int] = (5, 10),
    mrr_k: int = 10,
    ndcg_k: int = 10,
    hit_k: int = 10,
) -> dict:
    report = {}
    for k in recall_ks:
        metric = metrics.get(k) or metrics.get(str(k))
        report[f"Recall@{k}"] = None if metric is None else metric.get("recall")

    mrr_metric = metrics.get(mrr_k) or metrics.get(str(mrr_k))
    ndcg_metric = metrics.get(ndcg_k) or metrics.get(str(ndcg_k))
    hit_metric = metrics.get(hit_k) or metrics.get(str(hit_k))
    report["MRR"] = None if mrr_metric is None else mrr_metric.get("mrr")
    report[f"nDCG@{ndcg_k}"] = None if ndcg_metric is None else ndcg_metric.get("ndcg")
    report[f"Hit Rate@{hit_k}"] = None if hit_metric is None else hit_metric.get("hit")
    report["Latency"] = latency_seconds
    return report


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
        f"MRR={metric['mrr']:.3f} "
        f"nDCG={metric['ndcg']:.3f}"
    )


def format_report_metric_line(prefix: str, metrics: dict) -> str:
    def fmt(value):
        return "-" if value is None else f"{value:.3f}"

    return (
        f"{prefix} "
        f"Recall@5={fmt(metrics.get('Recall@5'))} "
        f"Recall@10={fmt(metrics.get('Recall@10'))} "
        f"MRR={fmt(metrics.get('MRR'))} "
        f"nDCG@10={fmt(metrics.get('nDCG@10'))} "
        f"HitRate@10={fmt(metrics.get('Hit Rate@10'))} "
        f"Latency={fmt(metrics.get('Latency'))}s"
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
                    "ndcg": metric.get("ndcg"),
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
        "ndcg",
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


def _report_rows(summary: dict) -> List[dict]:
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
        "rerank": summary.get("rerank"),
    }
    for stage_name, metric_group in [
        ("candidate", summary.get("candidate_report_metrics", {})),
        ("final", summary.get("final_report_metrics", {})),
    ]:
        rows.append(
            {
                **common,
                "stage": stage_name,
                "Recall@5": metric_group.get("Recall@5"),
                "Recall@10": metric_group.get("Recall@10"),
                "MRR": metric_group.get("MRR"),
                "nDCG@10": metric_group.get("nDCG@10"),
                "Hit Rate@10": metric_group.get("Hit Rate@10"),
                "Latency": metric_group.get("Latency"),
            }
        )
    return rows


def write_report_metrics_csv(path: str, summaries: Sequence[dict]) -> None:
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
        "rerank",
        "stage",
        "Recall@5",
        "Recall@10",
        "MRR",
        "nDCG@10",
        "Hit Rate@10",
        "Latency",
    ]
    rows = []
    for summary in summaries:
        rows.extend(_report_rows(summary))

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
