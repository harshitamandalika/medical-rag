from __future__ import annotations
import time
from dataclasses import dataclass

from retrieval.hybrid_retriever  import HybridRetriever
from generation.generator        import Generator, GenerationResult
from ingestion.embedder          import Embedder


@dataclass
class PipelineResult:
    question:          str
    answer:            str
    cited_pmids:       list[str]
    confidence:        str
    confidence_reason: str
    chunks:            list[dict]
    retrieval_meta:    dict
    generation_meta:   dict
    total_latency_ms:  float


class RAGPipeline:
    def __init__(self, k: int = 5, fetch_each: int = 20):
        print("Initializing RAG pipeline")
        t0 = time.time()

        shared_embedder = Embedder(show_progress=False)
        self.retriever  = HybridRetriever(embedder=shared_embedder)
        self.generator  = Generator()
        self.k          = k
        self.fetch_each = fetch_each

        print(f"Pipeline ready in {time.time() - t0:.1f}s")

    def run(
        self,
        question:  str,
        where:     dict | None = None,
    ) -> PipelineResult:
        t0 = time.time()

        chunks, retrieval_meta = self.retriever.retrieve(
            query=question,
            k=self.k,
            fetch_each=self.fetch_each,
            where=where,
        )

        gen_result: GenerationResult = self.generator.generate(
            question=question,
            chunks=chunks,
        )

        total_latency_ms = (time.time() - t0) * 1000

        generation_meta = {
            "generation_latency_ms": gen_result.latency_ms,
            "prompt_tokens":         gen_result.prompt_tokens,
            "completion_tokens":     gen_result.completion_tokens,
            "model":                 gen_result.model,
        }

        return PipelineResult(
            question          = question,
            answer            = gen_result.answer,
            cited_pmids       = gen_result.cited_pmids,
            confidence        = gen_result.confidence,
            confidence_reason = gen_result.confidence_reason,
            chunks            = chunks,
            retrieval_meta    = retrieval_meta,
            generation_meta   = generation_meta,
            total_latency_ms  = total_latency_ms,
        )