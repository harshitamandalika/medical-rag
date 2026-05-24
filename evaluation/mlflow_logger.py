from __future__ import annotations
import os
import mlflow

from pipeline.rag_pipeline   import PipelineResult
from evaluation.ragas_evaluator import RAGASResult

MLFLOW_EXPERIMENT = "pubmed_rag_eval"
PROMPT_VERSION    = "v1"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"


def _get_or_create_experiment() -> str:
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "./mlflow_runs")
    )
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        return mlflow.create_experiment(MLFLOW_EXPERIMENT)
    return experiment.experiment_id


def log_run(
    result:        PipelineResult,
    ragas_result:  RAGASResult | None = None,
    condition:     str | None = None,
    source:        str = "live",
) -> str:
    experiment_id = _get_or_create_experiment()

    with mlflow.start_run(experiment_id=experiment_id) as run:

        # Parameters
        mlflow.log_params({
            "question":        result.question[:250],
            "condition_filter": condition or "none",
            "k":               result.retrieval_meta.get("n_chunks_retrieved"),
            "model":           result.generation_meta.get("model"),
            "embedding_model": EMBEDDING_MODEL,
            "prompt_version":  PROMPT_VERSION,
            "source":          source,
        })

        # Retrieval metrics
        mlflow.log_metrics({
            "retrieval_latency_ms":  result.retrieval_meta.get("retrieval_latency_ms", 0),
            "dense_latency_ms":      result.retrieval_meta.get("dense_latency_ms", 0),
            "sparse_latency_ms":     result.retrieval_meta.get("sparse_latency_ms", 0),
            "n_chunks_retrieved":    result.retrieval_meta.get("n_chunks_retrieved", 0),
            "top1_similarity":       result.retrieval_meta.get("top1_similarity") or 0,
            "top1_rrf_score":        result.retrieval_meta.get("top1_rrf_score") or 0,
        })

        # Generation metrics
        mlflow.log_metrics({
            "generation_latency_ms": result.generation_meta.get("generation_latency_ms", 0),
            "prompt_tokens":         result.generation_meta.get("prompt_tokens", 0),
            "completion_tokens":     result.generation_meta.get("completion_tokens", 0),
            "total_latency_ms":      result.total_latency_ms,
        })

        # RAGAS metrics
        if ragas_result is not None:
            ragas_dict = {k: v for k, v in ragas_result.to_dict().items() if v is not None}
            if ragas_dict:
                mlflow.log_metrics(ragas_dict)

        # Tags
        mlflow.set_tags({
            "confidence":   result.confidence,
            "cited_pmids":  ",".join(result.cited_pmids),
            "source":       source,
        })

        print(f"  mlflow: logged run {run.info.run_id[:8]} to experiment '{MLFLOW_EXPERIMENT}'")
        return run.info.run_id


def fetch_metric_history(metric_name: str, n: int = 50) -> list[dict]:
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "./mlflow_runs")
    )
    client     = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=n,
    )

    history = []
    for run in runs:
        value = run.data.metrics.get(metric_name)
        if value is not None:
            history.append({
                "run_id":    run.info.run_id[:8],
                "value":     value,
                "timestamp": run.info.start_time,
                "question":  run.data.params.get("question", ""),
            })

    return list(reversed(history))   # chronological order for trend charts