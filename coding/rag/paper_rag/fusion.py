from typing import Dict, List, Mapping, Optional, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[dict]],
    top_k: int,
    rrf_k: int = 60,
    weights: Optional[Mapping[str, float]] = None,
) -> List[dict]:
    """Fuse ranked retrieval results using Reciprocal Rank Fusion.

    Ground-truth labels are not used here. Every candidate returned by any branch
    receives an RRF score based on its rank in each branch.
    """

    scores: Dict[int, float] = {}
    branch_ranks: Dict[int, Dict[str, int]] = {}
    branch_scores: Dict[int, Dict[str, float]] = {}

    for branch_name, hits in rankings.items():
        branch_weight = 1.0 if weights is None else weights.get(branch_name, 1.0)
        for rank, hit in enumerate(hits, start=1):
            paper_id = int(hit["paper_id"])
            scores[paper_id] = scores.get(paper_id, 0.0) + branch_weight / (rrf_k + rank)
            branch_ranks.setdefault(paper_id, {})[branch_name] = rank
            branch_scores.setdefault(paper_id, {})[branch_name] = float(hit.get("score", 0.0))

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        {
            "paper_id": paper_id,
            "score": float(scores[paper_id]),
            "branch_ranks": branch_ranks.get(paper_id, {}),
            "branch_scores": branch_scores.get(paper_id, {}),
        }
        for paper_id in ranked_ids
    ]
