from __future__ import annotations
import chromadb
from chromadb.config import Settings
from ingestion.chunker import Chunk


COLLECTION_NAME = "pubmed_abstracts"
CHROMA_PATH     = "./chroma_db"


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    collection: chromadb.Collection,
    chunks:     list[Chunk],
    vectors:    list[list[float]],
    batch_size: int = 500,
) -> None:
    assert len(chunks) == len(vectors), "chunks and vectors must be the same length"

    total   = len(chunks)
    written = 0

    for start in range(0, total, batch_size):
        end   = min(start + batch_size, total)
        batch_chunks  = chunks[start:end]
        batch_vectors = vectors[start:end]

        ids       = [c.chunk_id for c in batch_chunks]
        documents = [c.text     for c in batch_chunks]
        metadatas = [_chunk_to_metadata(c) for c in batch_chunks]

        collection.upsert(
            ids        = ids,
            embeddings = batch_vectors,
            documents  = documents,
            metadatas  = metadatas,
        )
        written += len(batch_chunks)
        print(f"  chroma: upserted {written}/{total} chunks")

    print(f"  chroma: collection now has {collection.count()} total chunks")


def _chunk_to_metadata(chunk: Chunk) -> dict:
    return {
        "pmid":       chunk.pmid,
        "title":      chunk.title[:200],
        "journal":    chunk.journal,
        "year":       chunk.year if chunk.year is not None else 0,
        "condition":  chunk.condition,
        "chunk_idx":  chunk.chunk_idx,
        "doi":        chunk.doi or "",
        "mesh_terms": " | ".join(chunk.mesh_terms),
    }


def query_collection(
    collection:   chromadb.Collection,
    query_vector: list[float],
    n_results:    int  = 5,
    where:        dict | None = None,
) -> list[dict]:
    kwargs: dict = {
        "query_embeddings": [query_vector],
        "n_results":        n_results,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    results = []
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append({
            "text":       doc,
            "metadata":   meta,
            "distance":   dist,
            "similarity": 1 - dist,
        })

    return results


def collection_stats(collection: chromadb.Collection) -> dict:
    count  = collection.count()
    sample = collection.peek(limit=min(count, 1000))
    conditions: dict[str, int] = {}
    for meta in sample["metadatas"]:
        cond = meta.get("condition", "unknown")
        conditions[cond] = conditions.get(cond, 0) + 1

    return {
        "total_chunks":          count,
        "conditions_in_sample":  conditions,
    }