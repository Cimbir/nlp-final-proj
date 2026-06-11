import math


def recall_at_k(results: list, qrels: dict, k: int) -> float:
    """
    Fraction of queries for which at least one relevant chunk appears in the top-K results
    """
    hits = 0
    for query_id, retrieved in results:
        relevant = set(qrels.get(query_id, []))
        if not relevant:
            continue
        if any(r in relevant for r in retrieved[:k]):
            hits += 1
    return hits / len(results) if results else 0.0


def mrr_at_k(results: list, qrels: dict, k: int = 10) -> float:
    """
    Mean Reciprocal Rank: average of 1/rank_of_first_relevant_chunk
    0 if no relevant chunk found in top-K
    """
    rr_sum = 0.0
    for query_id, retrieved in results:
        relevant = set(qrels.get(query_id, []))
        for rank, chunk_id in enumerate(retrieved[:k], start=1):
            if chunk_id in relevant:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(results) if results else 0.0


def ndcg_at_k(results: list, qrels: dict, k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain at K
    Binary relevance: 1 if chunk is relevant, 0 otherwise
    """
    ndcg_sum = 0.0
    for query_id, retrieved in results:
        relevant = set(qrels.get(query_id, []))
        if not relevant:
            continue

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, chunk_id in enumerate(retrieved[:k], start=1)
            if chunk_id in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

        ndcg_sum += dcg / idcg if idcg > 0 else 0.0

    return ndcg_sum / len(results) if results else 0.0


def mean_rank(results: list, qrels: dict) -> float:
    """
    Average rank of the first relevant chunk across all queries
    Returns inf if no relevant chunk was found in any retrieved list
    """
    rank_sum, found = 0.0, 0
    for query_id, retrieved in results:
        relevant = set(qrels.get(query_id, []))
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant:
                rank_sum += rank
                found += 1
                break
    return rank_sum / found if found else float("inf")


def run_evaluation(
    retriever,
    eval: list[dict],
    k_values: list[int] = [1, 5, 10],
    rank_k: int = 100,
    **retriever_kwargs,
) -> dict:
    """
    Run a retriever over all eval queries and compute all metrics

    Args:
        retriever: object with .search(query, k, **kwargs) -> [(score, text, chunk_id)]
        eval: list of {"query_id": str, "query": str, "chunk_id": str}
        k_values: list of K values to evaluate at
        rank_k: how many results to retrieve for MeanRank (should be >= corpus size for true mean rank)
        retriever_kwargs: extra keyword args forwarded to retriever.search()

    Returns:
        dict of metric_name -> float, e.g. {"Recall@1": 0.42, "MRR@10": 0.55, "MeanRank": 12.3, ...}
    """
    qrels: dict = {}
    for item in eval:
        qid = item["query_id"]
        qrels.setdefault(qid, []).append(item["chunk_id"])

    fetch_k = max(max(k_values), rank_k)
    results = []
    for item in eval:
        qid = item["query_id"]
        query = item["query"]
        retrieved = retriever.search(query, k=fetch_k, **retriever_kwargs)
        chunk_id_results = [r[2] for r in retrieved]
        results.append((qid, chunk_id_results))

    metrics: dict = {}
    for k in k_values:
        metrics[f"Recall@{k}"] = recall_at_k(results, qrels, k)
        metrics[f"MRR@{k}"] = mrr_at_k(results, qrels, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(results, qrels, k)
    metrics["MeanRank"] = mean_rank(results, qrels)

    return metrics


def print_metrics(metrics: dict, name: str = "Retriever") -> None:
    print(f"{name}")
    for metric, value in sorted(metrics.items()):
        print(f"{metric:<12} {value:.4f}")
