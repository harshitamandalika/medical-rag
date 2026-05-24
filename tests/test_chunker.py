from ingestion.chunker import chunk_record, chunk_records


DUMMY_RECORD = {
    "pmid":       "99999999",
    "title":      "Test Article",
    "abstract":   " ".join(["word"] * 300),
    "authors":    [],
    "journal":    "Test Journal",
    "year":       2024,
    "mesh_terms": [],
    "doi":        None,
}


def test_chunk_record_produces_chunks():
    chunks = chunk_record(DUMMY_RECORD, condition="test", chunk_size=250, chunk_overlap=50)
    assert len(chunks) >= 1


def test_chunk_ids_unique():
    chunks = chunk_record(DUMMY_RECORD, condition="test", chunk_size=250, chunk_overlap=50)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_metadata():
    chunks = chunk_record(DUMMY_RECORD, condition="type2_diabetes")
    assert chunks[0].pmid == "99999999"
    assert chunks[0].condition == "type2_diabetes"


def test_empty_abstract_returns_no_chunks():
    record = DUMMY_RECORD.copy()
    record["abstract"] = ""
    chunks = chunk_record(record, condition="test")
    assert chunks == []


def test_chunk_records_skips_empty():
    records = [DUMMY_RECORD, {**DUMMY_RECORD, "abstract": ""}]
    chunks = chunk_records(records, condition="test")
    assert len(chunks) >= 1