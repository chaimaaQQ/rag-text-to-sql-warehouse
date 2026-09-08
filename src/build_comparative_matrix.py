"""Build the final, traceable comparative matrix required by the PFA brief.

The main BIRD study and document-retrieval studies intentionally use
different corpora. This script keeps those values separate so that a
Knowledge Recall result is never misrepresented as a BIRD end-to-end metric.
"""

import argparse
import csv
import json
from pathlib import Path


PIPELINES = ("A", "B", "C", "D")
PIPELINE_LABELS = {
    "A": "A - Sans RAG",
    "B": "B - RAG schema seul",
    "C": "C - RAG metier seul",
    "D": "D - RAG hybride",
}


def load_json_safe(path: str | Path | None):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def nested_metric(data: dict | None, *keys: str):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, (float, int)) else None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def structural_metrics(structural: list[dict] | None) -> dict:
    if not structural:
        return {
            "valid_sql_rate": None,
            "refusal_rate": None,
            "static_structural_error_rate": None,
            "hallucinated_table_rate": None,
            "hallucinated_column_rate": None,
        }

    total = len(structural)
    attempted = [row for row in structural if row.get("error_type") != "empty_sql"]
    valid = sum(bool(row.get("is_valid")) for row in structural)
    structural_errors = sum(not row.get("is_valid") for row in attempted)
    table_hallucinations = sum(row.get("error_type") == "table_not_found" for row in structural)
    column_hallucinations = sum(
        row.get("error_type") in {"column_not_found", "column_unresolved"}
        for row in structural
    )
    return {
        "valid_sql_rate": rate(valid, total),
        "refusal_rate": rate(total - len(attempted), total),
        "static_structural_error_rate": rate(structural_errors, len(attempted)),
        "hallucinated_table_rate": rate(table_hallucinations, total),
        "hallucinated_column_rate": rate(column_hallucinations, total),
    }


def execution_error_metrics(classification: list[dict] | None) -> dict:
    """Report execution errors separately from static SQL validation and
    manual semantic annotation."""
    if not classification:
        return {
            "execution_structural_error_rate": None,
            "wrong_execution_result_rate": None,
            "execution_attempted": None,
        }

    excluded = {"insufficient_context", "empty_sql"}
    structural = {
        "unknown_column", "unknown_table", "ambiguous_column",
        "syntax_error", "unsupported_function", "other_execution_error",
    }
    attempted = [row for row in classification if row.get("category") not in excluded]
    return {
        "execution_structural_error_rate": rate(
            sum(row.get("category") in structural for row in attempted), len(attempted)
        ),
        "wrong_execution_result_rate": rate(
            sum(row.get("category") == "wrong_execution_result" for row in attempted), len(attempted)
        ),
        "execution_attempted": len(attempted),
    }


def heuristic_semantic_rate(rows: list[dict] | None) -> float | None:
    if not rows:
        return None
    evaluated = [row for row in rows if row.get("is_semantic_error") is not None]
    return rate(sum(bool(row.get("is_semantic_error")) for row in evaluated), len(evaluated))


def build_row(pipeline: str, results_dir: Path, evaluation_dir: Path,
              error_dir: Path) -> dict:
    execution = load_json_safe(results_dir / f"pipeline_{pipeline}" / "execution_accuracy.json")
    structural = load_json_safe(evaluation_dir / f"structural_{pipeline}.json")
    semantic = load_json_safe(error_dir / f"semantic_{pipeline}.json")
    execution_based = load_json_safe(error_dir / f"execution_based_{pipeline}.json")

    row = {
        "pipeline": PIPELINE_LABELS[pipeline],
        "execution_accuracy": nested_metric(execution, "summary", "execution_accuracy"),
        "n_comparable": nested_metric(execution, "summary", "n_comparable"),
        "heuristic_semantic_error_rate": heuristic_semantic_rate(semantic),
    }
    row.update(structural_metrics(structural))
    row.update(execution_error_metrics(execution_based))
    return row


def add_retrieval_metrics(rows: list[dict], schema_ablation: dict | None,
                          business_ablation: dict | None,
                          bird_ablation: dict | None) -> None:
    schema_method = schema_ablation.get("best_method_table_level") if schema_ablation else None
    table_recall = nested_metric(schema_ablation, "results", "table", schema_method, "recall@5")
    column_recall = nested_metric(schema_ablation, "results", "column", schema_method, "recall@5")

    business_method = business_ablation.get("best_method") if business_ablation else None
    business_recall = nested_metric(business_ablation, "results", business_method, "recall@5")
    bird_method = bird_ablation.get("best_method") if bird_ablation else None
    bird_recall = nested_metric(bird_ablation, "results", bird_method, "recall@5")

    for row, pipeline in zip(rows, PIPELINES):
        row["table_recall_at_5"] = table_recall if pipeline in {"B", "D"} else None
        row["column_recall_at_5"] = column_recall if pipeline in {"B", "D"} else None
        row["knowledge_recall_at_5_tpcds"] = business_recall if pipeline in {"C", "D"} else None
        row["knowledge_recall_at_5_bird_evidence"] = bird_recall if pipeline in {"C", "D"} else None


def build_matrix(results_dir: str, evaluation_dir: str, error_dir: str,
                 schema_ablation_file: str, business_ablation_file: str | None,
                 bird_ablation_file: str | None) -> list[dict]:
    rows = [
        build_row(pipeline, Path(results_dir), Path(evaluation_dir), Path(error_dir))
        for pipeline in PIPELINES
    ]
    add_retrieval_metrics(
        rows,
        load_json_safe(schema_ablation_file),
        load_json_safe(business_ablation_file),
        load_json_safe(bird_ablation_file),
    )
    return rows


def display_value(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.1%}"
    return str(value)


def print_markdown_table(rows: list[dict]) -> None:
    headers = (
        ("pipeline", "Pipeline"),
        ("execution_accuracy", "Execution Accuracy"),
        ("table_recall_at_5", "Table Recall@5"),
        ("column_recall_at_5", "Column Recall@5"),
        ("knowledge_recall_at_5_tpcds", "Knowledge Recall@5 (TPC-DS)*"),
        ("knowledge_recall_at_5_bird_evidence", "Knowledge Recall@5 (BIRD-Evidence)*"),
        ("valid_sql_rate", "Valid SQL Rate"),
        ("refusal_rate", "Taux de refus"),
        ("execution_structural_error_rate", "Erreurs structurelles (execution)"),
        ("wrong_execution_result_rate", "Resultats executes incorrects"),
    )
    print("| " + " | ".join(label for _, label in headers) + " |")
    print("|" + "---|" * len(headers))
    for row in rows:
        print("| " + " | ".join(display_value(row[key]) for key, _ in headers) + " |")
    print("\n*Les recalls documentaires sont des evaluations secondaires sur TPC-DS "
          "et BIRD-Evidence-Corpus, jamais un retrieval sur BIRD directement.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit la matrice comparative finale.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--evaluation_dir", required=True)
    parser.add_argument("--error_classification_dir", required=True)
    parser.add_argument("--schema_ablation_file", required=True)
    parser.add_argument("--business_ablation_file", default=None)
    parser.add_argument("--bird_ablation_file", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    rows = build_matrix(
        args.results_dir, args.evaluation_dir, args.error_classification_dir,
        args.schema_ablation_file, args.business_ablation_file, args.bird_ablation_file,
    )
    print_markdown_table(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "comparative_matrix.json", "w", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False, indent=2)
    with open(output_dir / "comparative_matrix.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
