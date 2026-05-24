from __future__ import annotations
import time

from ingestion.chroma_store import get_client, get_or_create_collection, query_collection
from ingestion.embedder import Embedder


class DenseRetriever:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder   = embedder or Embedder(show_progress=False)
        client          = get_client()
        self.collection = get_or_create_collection(client)

    def retrieve(
        self,
        query: str,
        k:     int        = 5,
        where: dict | None = None,
    ) -> tuple[list[dict], float]:
        t0           = time.time()
        query_vector = self.embedder.embed_query(query)
        results      = query_collection(self.collection, query_vector, n_results=k, where=where)
        latency_ms   = (time.time() - t0) * 1000

        print(f"  dense retriever: {len(results)} results in {latency_ms:.1f}ms")

        for rank, result in enumerate(results):
            result["rank"]      = rank + 1
            result["retriever"] = "dense"

        return results, latency_ms