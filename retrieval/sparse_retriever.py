from __future__ import annotations
import time
from rank_bm25 import BM25Okapi

from ingestion.chroma_store import get_client, get_or_create_collection


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class SparseRetriever:
    def __init__(self):
        print("  sparse retriever: loading corpus from ChromaDB to build BM25 index")
        t0 = time.time()

        client     = get_client()
        collection = get_or_create_collection(client)
        total      = collection.count()

        if total == 0:
            raise RuntimeError(
                "ChromaDB collection is empty. Run ingest_pipeline.py first."
            )

        raw = collection.get(include=["documents", "metadatas"])

        self.ids       = raw["ids"]
        self.documents = raw["documents"]
        self.metadatas = raw["metadatas"]

        tokenized      = [_tokenize(doc) for doc in self.documents]
        self.bm25      = BM25Okapi(tokenized)

        elapsed = time.time() - t0
        print(
            f"  sparse retriever: BM25 index built over {total} chunks in {elapsed:.1f}s"
        )

    def retrieve(
        self,
        query: str,
        k:     int        = 5,
        where: dict | None = None,
    ) -> tuple[list[dict], float]:
        t0          = time.time()
        query_tokens = _tokenize(query)
        scores       = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.ids, self.documents, self.metadatas, scores),
            key=lambda x: x[3],
            reverse=True,
        )

        if where:
            ranked = [
                item for item in ranked
                if _matches_filter(item[2], where)
            ]

        top_k      = ranked[:k]
        latency_ms = (time.time() - t0) * 1000

        print(f"  sparse retriever: {len(top_k)} results in {latency_ms:.1f}ms")

        results = []
        for rank, (chunk_id, text, metadata, score) in enumerate(top_k):
            results.append({
                "text":      text,
                "metadata":  metadata,
                "score":     float(score),
                "rank":      rank + 1,
                "retriever": "sparse",
            })

        return results, latency_ms


def _matches_filter(metadata: dict, where: dict) -> bool:
    for key, value in where.items():
        meta_val = metadata.get(key)
        if isinstance(value, dict):
            op, operand = next(iter(value.items()))
            if op == "$eq"  and meta_val != operand:
                return False
            if op == "$ne"  and meta_val == operand:
                return False
            if op == "$gt"  and not (meta_val is not None and meta_val >  operand):
                return False
            if op == "$gte" and not (meta_val is not None and meta_val >= operand):
                return False
            if op == "$lt"  and not (meta_val is not None and meta_val <  operand):
                return False
            if op == "$lte" and not (meta_val is not None and meta_val <= operand):
                return False
        else:
            if meta_val != value:
                return False
    return True