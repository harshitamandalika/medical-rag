from pipeline.rag_pipeline import PipelineResult


def test_pipeline_result_fields():
    result = PipelineResult(
        question         = "Test question",
        answer           = "Test answer",
        cited_pmids      = ["12345678"],
        confidence       = "high",
        confidence_reason= "Test reason",
        chunks           = [],
        retrieval_meta   = {"retrieval_latency_ms": 100, "n_chunks_retrieved": 5},
        generation_meta  = {"generation_latency_ms": 200},
        total_latency_ms = 300,
    )

    assert result.question == "Test question"
    assert result.confidence == "high"
    assert result.cited_pmids == ["12345678"]
    assert result.total_latency_ms == 300