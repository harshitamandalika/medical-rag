from __future__ import annotations
import time

from retrieval.dense_retriever  import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from ingestion.embedder         import Embedder

RRF_K = 60

class HybridRetriever:
    def __init__(self, embedder: Embedder | None = None):
        shared_embedder   = embedder or Embedder(show_progress=False)
        self.dense        = DenseRetriever(embedder=shared_embedder)
        self.sparse       = SparseRetriever()

    def retrieve(
        self,
        query:        str,
        k:            int        = 5,
        where:        dict | None = None,
        fetch_each:   int        = 20,
    ) -> tuple[list[dict], dict]:
        t0 = time.time()

        dense_results,  dense_latency_ms  = self.dense.retrieve(query,  k=fetch_each, where=where)
        sparse_results, sparse_latency_ms = self.sparse.retrieve(query, k=fetch_each, where=where)

        fused      = _reciprocal_rank_fusion(dense_results, sparse_results, k=k)
        latency_ms = (time.time() - t0) * 1000

        print(
            f"  hybrid retriever: {len(fused)} results in {latency_ms:.1f}ms "
            f"(dense {dense_latency_ms:.0f}ms, sparse {sparse_latency_ms:.0f}ms)"
        )

        retrieval_metadata = {
            "retrieval_latency_ms":        latency_ms,
            "dense_latency_ms":            dense_latency_ms,
            "sparse_latency_ms":           sparse_latency_ms,
            "top1_similarity":             fused[0]["similarity"] if fused else None,
            "top1_rrf_score":              fused[0]["rrf_score"]  if fused else None,
            "n_chunks_retrieved":          len(fused),
            "n_dense_candidates":          len(dense_results),
            "n_sparse_candidates":         len(sparse_results),
        }

        return fused, retrieval_metadata


def _reciprocal_rank_fusion(
    dense_results:  list[dict],
    sparse_results: list[dict],
    k:              int = 5,
    rrf_k:          int = RRF_K,
) -> list[dict]:
    scores:  dict[str, float] = {}
    records: dict[str, dict]  = {}

    for ranked_list in (dense_results, sparse_results):
        for result in ranked_list:
            chunk_id = result["metadata"]["pmid"] + "_chunk_" + str(result["metadata"]["chunk_idx"])
            rrf_score = 1.0 / (rrf_k + result["rank"])

            scores[chunk_id]  = scores.get(chunk_id, 0.0) + rrf_score

            if chunk_id not in records:
                records[chunk_id] = result.copy()
                records[chunk_id]["sources"] = [result["retriever"]]
            else:
                if result["retriever"] not in records[chunk_id]["sources"]:
                    records[chunk_id]["sources"].append(result["retriever"])

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_k  = ranked[:k]

    fused = []
    for final_rank, (chunk_id, rrf_score) in enumerate(top_k):
        record               = records[chunk_id].copy()
        record["rrf_score"]  = round(rrf_score, 6)
        record["rank"]       = final_rank + 1
        record["retriever"]  = "hybrid"

        if "similarity" not in record:
            record["similarity"] = None
        fused.append(record)

    return fused