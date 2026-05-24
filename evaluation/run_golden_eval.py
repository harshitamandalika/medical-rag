from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from pipeline.rag_pipeline          import RAGPipeline, PipelineResult
from evaluation.ragas_evaluator     import evaluate_batch, RAGASResult
from evaluation.mlflow_logger       import log_run

GOLDEN_TEST_SET_PATH = Path("data/golden_test_set.json")


def load_golden_test_set(
    condition: str | None = None,
    n:         int | None = None,
) -> list[dict]:
    items = json.loads(GOLDEN_TEST_SET_PATH.read_text())
    if condition:
        items = [i for i in items if i["condition"] == condition]
    if n:
        items = items[:n]
    return items


def print_summary(
    items:         list[dict],
    ragas_results: list[RAGASResult],
) -> None:
    print("\nGOLDEN EVAL SUMMARY")

    conditions: dict[str, list[RAGASResult]] = {}
    for item, r in zip(items, ragas_results):
        cond = item["condition"]
        conditions.setdefault(cond, []).append(r)

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def fmt(v):
        return f"{v:.4f}" if v is not None else "n/a"

    header = f"{'condition':<25} {'faith':>7} {'ans_rel':>8} {'ctx_prec':>9} {'ctx_rec':>8} {'n':>4}"
    print(header)
    print("-" * len(header))

    all_results = []
    for cond, results in sorted(conditions.items()):
        all_results.extend(results)
        print(
            f"{cond:<25} "
            f"{fmt(mean([r.faithfulness      for r in results])):>7} "
            f"{fmt(mean([r.answer_relevancy  for r in results])):>8} "
            f"{fmt(mean([r.context_precision for r in results])):>9} "
            f"{fmt(mean([r.context_recall    for r in results])):>8} "
            f"{len(results):>4}"
        )

    print("-" * len(header))
    print(
        f"{'OVERALL':<25} "
        f"{fmt(mean([r.faithfulness      for r in all_results])):>7} "
        f"{fmt(mean([r.answer_relevancy  for r in all_results])):>8} "
        f"{fmt(mean([r.context_precision for r in all_results])):>9} "
        f"{fmt(mean([r.context_recall    for r in all_results])):>8} "
        f"{len(all_results):>4}"
    )

def run_golden_eval(
    condition: str | None = None,
    n:         int | None = None,
) -> None:
    items = load_golden_test_set(condition=condition, n=n)
    print(f"Loaded {len(items)} questions from golden test set")

    pipeline = RAGPipeline(k=5)

    print(f"\nRunning pipeline on {len(items)} questions")
    pipeline_results: list[PipelineResult] = []
    for i, item in enumerate(items):
        print(f"  [{i+1}/{len(items)}] {item['question'][:70]}...")
        where = {"condition": {"$eq": item["condition"]}} if item.get("condition") else None
        result = pipeline.run(item["question"], where=where)
        pipeline_results.append(result)
        time.sleep(2)

    print(f"\nRunning RAGAS batch evaluation on {len(pipeline_results)} results")
    ground_truths = [item["ground_truth"] for item in items]
    ragas_results = evaluate_batch(pipeline_results, ground_truths)

    print(f"\nLogging {len(pipeline_results)} runs to MLflow")
    for item, pipeline_result, ragas_result in zip(items, pipeline_results, ragas_results):
        log_run(
            result       = pipeline_result,
            ragas_result = ragas_result,
            condition    = item.get("condition"),
            source       = "golden_eval",
        )

    print_summary(items, ragas_results)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RAGAS eval over the golden test set")
    p.add_argument("--condition", type=str, default=None,
                   help="Evaluate only this condition (e.g. type2_diabetes)")
    p.add_argument("--n", type=int, default=None,
                   help="Evaluate only the first n questions (for smoke testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_golden_eval(condition=args.condition, n=args.n)