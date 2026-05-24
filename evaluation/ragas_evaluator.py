from __future__ import annotations
import os
import time
from dataclasses import dataclass

import nest_asyncio
nest_asyncio.apply()

import aiohttp
if not hasattr(aiohttp, "ClientConnectorDNSError"):
    aiohttp.ClientConnectorDNSError = aiohttp.ClientConnectorError

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from pipeline.rag_pipeline import PipelineResult


JUDGE_MODEL     = "models/gemini-2.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RAGASResult:
    faithfulness:       float | None
    answer_relevancy:   float | None
    context_precision:  float | None
    context_recall:     float | None

    def to_dict(self) -> dict:
        return {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }

    def display(self) -> None:
        print("RAGAS scores:")
        for k, v in self.to_dict().items():
            score = f"{v:.4f}" if v is not None else "n/a"
            print(f"  {k:<22} {score}")


def _build_ragas_llm() -> LangchainLLMWrapper:
    llm = ChatGoogleGenerativeAI(
        model=JUDGE_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
        max_retries=6,
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def _build_metrics(include_recall: bool, include_precision: bool = True) -> list:
    ragas_llm        = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    metrics = [faithfulness, answer_relevancy]

    if include_precision:
        metrics.append(context_precision)
    if include_recall:
        metrics.append(context_recall)

    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    return metrics


def _run_evaluate(dataset: Dataset, metrics: list) -> object:
    run_config = RunConfig(
        max_workers=1,
        timeout=180,
        max_retries=6,
    )
    return evaluate(dataset, metrics=metrics, run_config=run_config)


def evaluate_result(
    result:       PipelineResult,
    ground_truth: str | None = None,
) -> RAGASResult:
    contexts = [chunk["text"] for chunk in result.chunks]

    row = {
        "question": [result.question],
        "answer":   [result.answer],
        "contexts": [contexts],
    }
    if ground_truth is not None:
        row["ground_truth"] = [ground_truth]

    dataset = Dataset.from_dict(row)
    metrics = _build_metrics(
        include_recall    = ground_truth is not None,
        include_precision = ground_truth is not None,
    )
    scores  = _run_evaluate(dataset, metrics)
    df      = scores.to_pandas()

    def _get(col: str) -> float | None:
        if col in df.columns:
            val = df[col].iloc[0]
            return float(val) if val is not None else None
        return None

    return RAGASResult(
        faithfulness      = _get("faithfulness"),
        answer_relevancy  = _get("answer_relevancy"),
        context_precision = _get("context_precision"),
        context_recall    = _get("context_recall"),
    )


def evaluate_batch(
    results:       list[PipelineResult],
    ground_truths: list[str],
) -> list[RAGASResult]:
    assert len(results) == len(ground_truths), "results and ground_truths must be same length"

    ragas_results = []
    for i, (result, ground_truth) in enumerate(zip(results, ground_truths)):
        print(f"  ragas: evaluating result {i+1}/{len(results)}")
        r = evaluate_result(result, ground_truth=ground_truth)
        r.display()
        ragas_results.append(r)
        if i < len(results) - 1:
            print("  ragas: waiting 20s before next eval")
            time.sleep(20)

    return ragas_results