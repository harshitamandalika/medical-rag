from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from ingestion.pubmed_client import PubMedClient
from ingestion.parser import parse_pubmed_xml
from ingestion.chunker import chunk_records
from ingestion.embedder import Embedder
from ingestion.chroma_store import (
    get_client,
    get_or_create_collection,
    upsert_chunks,
    collection_stats,
)

CONDITIONS: list[tuple[str, str, int]] = [
    ("Type 2 Diabetes treatment",         "type2_diabetes",  100),
    ("hypertension management",           "hypertension",    100),
    ("asthma treatment clinical trial",   "asthma",          100),
    ("heart failure therapy",             "heart_failure",   100),
    ("depression pharmacotherapy",        "depression",      100),
    ("chronic kidney disease management", "chronic_kidney",   50),
]

DATA_DIR = Path("data")


def run_ingestion(
    conditions:  list[tuple[str, str, int]],
    save_parsed: bool = True,
) -> None:
    print("PubMed RAG Ingestion Pipeline")

    t_total    = time.time()
    client     = get_client()
    collection = get_or_create_collection(client)
    embedder   = Embedder()

    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "parsed").mkdir(parents=True, exist_ok=True)

    pubmed               = PubMedClient()
    total_chunks_written = 0

    for query, condition_tag, n in conditions:
        print(f"\nCondition: {condition_tag}  query: '{query}'  n={n}")

        search  = pubmed.search(query, n)
        xml_str = pubmed.fetch_xml(search["query_key"], search["webenv"], n)

        raw_path = DATA_DIR / "raw" / f"{condition_tag}.xml"
        raw_path.write_text(xml_str, encoding="utf-8")
        print(f"  saved raw XML to {raw_path}")

        records = parse_pubmed_xml(xml_str)
        print(f"  parsed {len(records)} records")

        if save_parsed:
            parsed_path = DATA_DIR / "parsed" / f"{condition_tag}.json"
            parsed_path.write_text(
                json.dumps(records, indent=2, ensure_ascii=False)
            )
            print(f"  saved parsed JSON to {parsed_path}")

        chunks = chunk_records(records, condition=condition_tag)

        if not chunks:
            print(f"  WARNING: no chunks produced for {condition_tag}, skipping")
            continue

        texts   = [c.text for c in chunks]
        vectors = embedder.embed(texts)

        upsert_chunks(collection, chunks, vectors)
        total_chunks_written += len(chunks)

        time.sleep(1)

    elapsed = time.time() - t_total
    stats   = collection_stats(collection)

    print("\nINGESTION COMPLETE")
    print(f"  Total time     : {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Chunks written : {total_chunks_written}")
    print(f"  ChromaDB total : {stats['total_chunks']} chunks")
    print("\n  Condition breakdown (from sample):")
    for cond, count in sorted(stats["conditions_in_sample"].items()):
        print(f"    {cond:<30} {count} chunks")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PubMed RAG ingestion pipeline")
    p.add_argument(
        "--condition", type=str, default=None,
        help="Run only this condition tag (e.g. 'hypertension'). Default: all."
    )
    p.add_argument(
        "--n", type=int, default=None,
        help="Override n_abstracts for the selected condition."
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Skip saving parsed JSON to data/parsed/ to save disk space."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.condition:
        targets = [
            (q, tag, args.n or n)
            for q, tag, n in CONDITIONS
            if tag == args.condition
        ]
        if not targets:
            print(f"ERROR: condition '{args.condition}' not found in CONDITIONS list.")
            print(f"Available: {[tag for _, tag, _ in CONDITIONS]}")
            raise SystemExit(1)
    else:
        targets = [
            (q, tag, args.n or n)
            for q, tag, n in CONDITIONS
        ]

    run_ingestion(targets, save_parsed=not args.no_save)