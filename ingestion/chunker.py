from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Chunk:
    chunk_id:   str
    pmid:       str
    text:       str
    chunk_idx:  int
    title:      str
    journal:    str
    year:       int | None
    condition:  str    
    mesh_terms: list[str]   = field(default_factory=list)
    doi:        str | None  = None


def _whitespace_tokens(text: str) -> list[str]:
    return text.split()


def chunk_record(
    record:        dict,
    condition:     str,
    chunk_size:    int = 250,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    abstract = record.get("abstract", "").strip()
    if not abstract:
        return []

    tokens = _whitespace_tokens(abstract)
    chunks: list[Chunk] = []

    step  = chunk_size - chunk_overlap
    start = 0
    idx   = 0

    while start < len(tokens):
        end        = min(start + chunk_size, len(tokens))
        chunk_text = " ".join(tokens[start:end])

        chunks.append(
            Chunk(
                chunk_id   = f"{record['pmid']}_chunk_{idx}",
                pmid       = record["pmid"],
                text       = chunk_text,
                chunk_idx  = idx,
                title      = record.get("title", ""),
                journal    = record.get("journal", ""),
                year       = record.get("year"),
                condition  = condition,
                mesh_terms = record.get("mesh_terms", []),
                doi        = record.get("doi"),
            )
        )

        if end == len(tokens):
            break

        start += step
        idx   += 1

    return chunks


def chunk_records(
    records:       list[dict],
    condition:     str,
    chunk_size:    int = 250,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    skipped = 0

    for record in records:
        result = chunk_record(record, condition, chunk_size, chunk_overlap)
        if result:
            all_chunks.extend(result)
        else:
            skipped += 1

    print(
        f"  chunker: {len(records)} records produced "
        f"{len(all_chunks)} chunks ({skipped} skipped, no abstract)"
    )
    return all_chunks