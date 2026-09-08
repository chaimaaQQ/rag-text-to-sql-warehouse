"""Run the mandatory factorial main study reproducibly.

It fixes the generator and retrieval settings, runs A/B/C/D three times by
default, and keeps each run isolated. Existing result files are never
overwritten unless ``--overwrite`` is explicitly passed.
"""

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean, pstdev

from classify_pipeline_errors import classify_generated_execution
from execution_accuracy import evaluate_pipeline
from pipeline_baseline import run_pipeline_a, save_results as save_a
from pipeline_business_rag import run_pipeline_c, save_results as save_c
from pipeline_hybrid_rag import run_pipeline_d, save_results as save_d
from pipeline_schema_rag import run_pipeline_b, save_results as save_b
from sql_validator import save_results as save_structural, validate_batch


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def existing_or_prepare(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} existe deja. Utilisez un autre --output_dir ou --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def error_rates(structural: list[dict], execution_categories: list[dict]) -> dict:
    attempted = [row for row in structural if row.get("error_type") != "empty_sql"]
    static_errors = sum(not row.get("is_valid") for row in attempted)
    excluded = {"insufficient_context", "empty_sql"}
    execution_attempted = [row for row in execution_categories if row["category"] not in excluded]
    execution_structural = {
        "unknown_column", "unknown_table", "ambiguous_column",
        "syntax_error", "unsupported_function", "other_execution_error",
    }
    return {
        "valid_sql_rate": sum(bool(row.get("is_valid")) for row in structural) / len(structural),
        "refusal_rate": sum(row.get("error_type") == "empty_sql" for row in structural) / len(structural),
        "static_structural_error_rate": static_errors / len(attempted) if attempted else None,
        "execution_structural_error_rate": (
            sum(row["category"] in execution_structural for row in execution_attempted)
            / len(execution_attempted) if execution_attempted else None
        ),
        "wrong_execution_result_rate": (
            sum(row["category"] == "wrong_execution_result" for row in execution_attempted)
            / len(execution_attempted) if execution_attempted else None
        ),
    }


def aggregate(values: list[float | None]) -> dict:
    usable = [value for value in values if value is not None]
    return {
        "mean": mean(usable) if usable else None,
        "std": pstdev(usable) if len(usable) > 1 else 0.0 if usable else None,
        "n_runs": len(usable),
    }


def summarise(run_metrics: dict[str, list[dict]], manifest: dict) -> dict:
    summary = {"manifest": manifest, "pipelines": {}}
    metric_names = (
        "execution_accuracy", "valid_sql_rate", "refusal_rate",
        "static_structural_error_rate", "execution_structural_error_rate",
        "wrong_execution_result_rate", "mean_llm_latency_seconds",
        "mean_prompt_tokens", "mean_completion_tokens",
    )
    for pipeline, metrics in run_metrics.items():
        summary["pipelines"][pipeline] = {
            metric: aggregate([run.get(metric) for run in metrics])
            for metric in metric_names
        }
    return summary


def run_one(pipeline: str, args, run_index: int, seed: int, output_dir: Path) -> dict:
    run_dir = output_dir / f"pipeline_{pipeline}" / f"run_{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    generated_path = run_dir / "generated_sql.json"

    common = {
        "model": args.model, "temperature": args.temperature,
        "limit": args.limit, "run_index": run_index, "seed": seed,
    }
    if pipeline == "A":
        generated = run_pipeline_a(args.questions, **common)
        save_a(generated, generated_path)
    elif pipeline == "B":
        generated = run_pipeline_b(
            args.questions, args.index_dir, args.schema_method, args.top_k, **common,
        )
        save_b(generated, generated_path)
    elif pipeline == "C":
        generated = run_pipeline_c(args.questions, top_k=args.top_k, **common)
        save_c(generated, generated_path)
    else:
        generated = run_pipeline_d(
            args.questions, args.index_dir, args.schema_method, args.top_k, **common,
        )
        save_d(generated, generated_path)

    structural = validate_batch(str(generated_path), args.schemas_dir)
    structural_path = run_dir / "structural_validation.json"
    save_structural(structural, structural_path)

    execution = evaluate_pipeline(
        str(generated_path), args.questions, args.databases_dir,
        args.round_decimals, args.timeout,
    )
    execution_path = run_dir / "execution_accuracy.json"
    write_json(execution_path, execution)

    categories = classify_generated_execution(generated_path, execution_path)
    category_path = run_dir / "execution_categories.json"
    write_json(category_path, categories)

    provenance = {
        "pipeline": pipeline, "run_index": run_index, "seed": seed,
        "model": args.model, "temperature": args.temperature,
        "top_k": args.top_k, "schema_retrieval_method": args.schema_method,
        "auto_correction": False,
    }
    write_json(run_dir / "provenance.json", provenance)

    generation_count = len(generated)
    rates = error_rates(structural, categories)
    return {
        "run_index": run_index,
        "seed": seed,
        "execution_accuracy": execution["summary"]["execution_accuracy"],
        "mean_llm_latency_seconds": (
            sum(row.get("latency_seconds", 0.0) for row in generated) / generation_count
            if generation_count else None
        ),
        "mean_prompt_tokens": (
            sum(row.get("prompt_tokens", 0) for row in generated) / generation_count
            if generation_count else None
        ),
        "mean_completion_tokens": (
            sum(row.get("completion_tokens", 0) for row in generated) / generation_count
            if generation_count else None
        ),
        **rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Etude principale A/B/C/D avec trois repetitions.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--schemas_dir", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--databases_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--schema_method", choices=("bm25", "embeddings", "hybrid"), default="hybrid")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--round_decimals", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.runs < 3:
        parser.error("Le cahier des charges impose au moins 3 repetitions (--runs >= 3).")

    output_dir = Path(args.output_dir)
    existing_or_prepare(output_dir, args.overwrite)
    manifest = {
        "questions": args.questions, "schemas_dir": args.schemas_dir,
        "index_dir": args.index_dir, "databases_dir": args.databases_dir,
        "model": args.model, "temperature": args.temperature, "top_k": args.top_k,
        "schema_method": args.schema_method, "auto_correction": False,
        "runs": args.runs, "seeds": list(range(args.seed_start, args.seed_start + args.runs)),
    }
    write_json(output_dir / "main_study_manifest.json", manifest)

    metrics = {pipeline: [] for pipeline in ("A", "B", "C", "D")}
    for run_index, seed in enumerate(manifest["seeds"]):
        for pipeline in metrics:
            print(f"\n=== Pipeline {pipeline}, run {run_index}, seed {seed} ===")
            metrics[pipeline].append(run_one(pipeline, args, run_index, seed, output_dir))
    write_json(output_dir / "main_study_runs.json", metrics)
    write_json(output_dir / "main_study_summary.json", summarise(metrics, manifest))


if __name__ == "__main__":
    main()
