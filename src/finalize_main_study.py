"""Finalize a manually resumed three-run A/B/C/D main study.

The standard main-study launcher performs these operations inline.  This
script provides the same post-processing when the generation runs were
executed individually with --resume.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

from classify_pipeline_errors import (
    EXCLUDED_CATEGORIES,
    SEMANTIC_CATEGORIES,
    STRUCTURAL_CATEGORIES,
    classify_generated_execution,
)
from hypothesis_h1_test_paired import run_mcnemar
from sql_validator import save_results as save_structural
from sql_validator import validate_batch


PIPELINES = ("A", "B", "C", "D")
RUN_NAMES = ("run_00", "run_01", "run_02")
PIPELINE_LABELS = {
    "A": "A - Sans RAG",
    "B": "B - RAG schema seul",
    "C": "C - RAG metier seul",
    "D": "D - RAG hybride",
}


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def rates(structural: list[dict], categories: list[dict]) -> dict:
    total = len(structural)
    valid_sql = sum(bool(row.get("is_valid")) for row in structural)
    refused = sum(row.get("error_type") == "empty_sql" for row in structural)
    attempted = [row for row in categories if row["category"] not in EXCLUDED_CATEGORIES]
    return {
        "valid_sql_rate": valid_sql / total if total else None,
        "refusal_rate": refused / total if total else None,
        "execution_structural_error_rate": (
            sum(row["category"] in STRUCTURAL_CATEGORIES for row in attempted) / len(attempted)
            if attempted else None
        ),
        "wrong_execution_result_rate": (
            sum(row["category"] in SEMANTIC_CATEGORIES for row in attempted) / len(attempted)
            if attempted else None
        ),
    }


def aggregate(values: list[float | None]) -> dict:
    usable = [value for value in values if value is not None]
    return {
        "mean": mean(usable) if usable else None,
        "std": pstdev(usable) if len(usable) > 1 else 0.0 if usable else None,
        "n_runs": len(usable),
        "values": values,
    }


def paired_h1(b_rows: list[dict], d_rows: list[dict]) -> dict:
    b = {row["question_id"]: row for row in b_rows}
    d = {row["question_id"]: row for row in d_rows}
    attempted_b = {qid for qid, row in b.items() if row["category"] not in EXCLUDED_CATEGORIES}
    attempted_d = {qid for qid, row in d.items() if row["category"] not in EXCLUDED_CATEGORIES}
    paired = sorted(attempted_b & attempted_d)

    structural_b = [b[qid]["category"] in STRUCTURAL_CATEGORIES for qid in paired]
    structural_d = [d[qid]["category"] in STRUCTURAL_CATEGORIES for qid in paired]
    semantic_b = [b[qid]["category"] in SEMANTIC_CATEGORIES for qid in paired]
    semantic_d = [d[qid]["category"] in SEMANTIC_CATEGORIES for qid in paired]
    structural = run_mcnemar(structural_b, structural_d)
    semantic = run_mcnemar(semantic_b, semantic_d)
    return {
        "n_paired": len(paired),
        "n_attempted_B": len(attempted_b),
        "n_attempted_D": len(attempted_d),
        "match_B": sum(b[qid]["category"] == "match" for qid in paired),
        "match_D": sum(d[qid]["category"] == "match" for qid in paired),
        "structural_test_paired": structural,
        "semantic_test_paired": semantic,
        "H1_supported_paired": bool(
            not structural["significant_at_0.05"]
            and semantic["significant_at_0.05"]
            and semantic["rate_D"] < semantic["rate_B"]
        ),
    }


def write_report_draft(path: Path, pipeline_summary: dict, h1: dict,
                       annotation_agreement: dict | None) -> None:
    lines = [
        "# Synthèse automatique — étude principale BIRD Mini-Dev",
        "",
        "## Résultats d'exécution (moyenne ± écart-type, 3 runs)",
        "",
        "| Pipeline | Execution Accuracy |",
        "| --- | --- |",
    ]
    for pipeline in PIPELINES:
        metric = pipeline_summary[pipeline]["summary"]["execution_accuracy"]
        lines.append(
            f"| {PIPELINE_LABELS[pipeline]} | {100 * metric['mean']:.2f} % ± {100 * metric['std']:.2f} % |"
        )

    lines.extend([
        "",
        "## H1 — comparaison appariée B vs D",
        "",
        "| Run | Erreurs sémantiques B | Erreurs sémantiques D | p sémantique | p structurelle | H1 selon le protocole par run |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for run_name, result in h1["per_run"].items():
        semantic = result["semantic_test_paired"]
        structural = result["structural_test_paired"]
        lines.append(
            f"| {run_name} | {100 * semantic['rate_B']:.1f} % | {100 * semantic['rate_D']:.1f} % | "
            f"{semantic['p_value']:.3g} | {structural['p_value']:.3g} | {result['H1_supported_paired']} |"
        )

    pooled = h1["pooled_question_run_pairs"]
    lines.extend([
        "",
        "## Limite d'interprétation",
        "",
        "Les trois runs montrent une réduction statistiquement significative des erreurs sémantiques de D. "
        "Cependant, le résumé pooled des paires question-run indique une hausse significative des erreurs "
        "structurelles de D ; il est descriptif car les mêmes questions sont répétées entre seeds. "
        "La conclusion finale doit donc présenter les trois tests par run et ne pas affirmer une amélioration structurelle.",
    ])
    if annotation_agreement:
        lines.extend([
            "",
            "## Annotation humaine",
            "",
            f"{annotation_agreement['n_completed']} cas ont été annotés. L'accord brut est de "
            f"{100 * annotation_agreement['raw_agreement']:.1f} % et Cohen's kappa est de "
            f"{annotation_agreement['cohens_kappa']:.3f}. Aucun désaccord n'a été enregistré.",
            "",
            "Cette valeur n'est interprétable comme un accord inter-annotateurs que si les deux colonnes "
            "ont effectivement été remplies de manière indépendante.",
        ])
    else:
        lines.extend([
            "",
            "## Annotation humaine requise",
            "",
            "La feuille `manual_annotation.csv` doit être complétée indépendamment par deux annotateurs, puis "
            "Cohen's kappa doit être calculé. Cette étape est nécessaire avant de considérer ce rapport comme final.",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalise les 12 runs A/B/C/D.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--schemas_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    run_metrics: dict[str, dict[str, dict]] = {pipeline: {} for pipeline in PIPELINES}
    categories_by_run: dict[str, dict[str, list[dict]]] = {pipeline: {} for pipeline in PIPELINES}

    for pipeline in PIPELINES:
        for run_name in RUN_NAMES:
            run_dir = results_dir / f"pipeline_{pipeline}" / run_name
            generated_path = run_dir / "generated_sql.json"
            execution_path = run_dir / "execution_accuracy.json"
            if not generated_path.exists() or not execution_path.exists():
                raise FileNotFoundError(f"Run incomplet : {run_dir}")

            structural = validate_batch(str(generated_path), args.schemas_dir)
            save_structural(structural, str(run_dir / "structural_validation.json"))
            categories = classify_generated_execution(generated_path, execution_path)
            write_json(run_dir / "execution_categories.json", categories)
            categories_by_run[pipeline][run_name] = categories

            execution = load_json(execution_path)["summary"]
            run_metrics[pipeline][run_name] = {
                "execution_accuracy": execution["execution_accuracy"],
                "n_match": execution["n_match"],
                "n_comparable": execution["n_comparable"],
                "n_total": execution["n_total"],
                **rates(structural, categories),
            }

    pipeline_summary = {}
    matrix = []
    metric_names = (
        "execution_accuracy", "valid_sql_rate", "refusal_rate",
        "execution_structural_error_rate", "wrong_execution_result_rate",
    )
    for pipeline in PIPELINES:
        per_run = run_metrics[pipeline]
        summary = {metric: aggregate([per_run[run][metric] for run in RUN_NAMES]) for metric in metric_names}
        pipeline_summary[pipeline] = {"runs": per_run, "summary": summary}
        matrix.append({
            "pipeline": PIPELINE_LABELS[pipeline],
            "execution_accuracy": summary["execution_accuracy"]["mean"],
            "execution_accuracy_std": summary["execution_accuracy"]["std"],
            "valid_sql_rate": summary["valid_sql_rate"]["mean"],
            "refusal_rate": summary["refusal_rate"]["mean"],
            "execution_structural_error_rate": summary["execution_structural_error_rate"]["mean"],
            "wrong_execution_result_rate": summary["wrong_execution_result_rate"]["mean"],
        })

    h1_by_run = {
        run_name: paired_h1(categories_by_run["B"][run_name], categories_by_run["D"][run_name])
        for run_name in RUN_NAMES
    }
    pooled_b = [row for run_name in RUN_NAMES for row in categories_by_run["B"][run_name]]
    pooled_d = [
        {**row, "question_id": f"{run_name}:{row['question_id']}"}
        for run_name in RUN_NAMES for row in categories_by_run["D"][run_name]
    ]
    pooled_b = [
        {**row, "question_id": f"{run_name}:{row['question_id']}"}
        for run_name in RUN_NAMES for row in categories_by_run["B"][run_name]
    ]
    h1 = {
        "per_run": h1_by_run,
        "all_runs_support_H1": all(result["H1_supported_paired"] for result in h1_by_run.values()),
        "pooled_question_run_pairs": paired_h1(pooled_b, pooled_d),
        "pooled_note": "Les paires question-run sont répétées entre seeds ; le résultat pooled est descriptif. Les résultats par run sont la référence principale.",
    }

    write_json(output_dir / "main_study_summary.json", {"pipelines": pipeline_summary})
    write_json(output_dir / "comparative_matrix.json", matrix)
    with open(output_dir / "comparative_matrix.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    write_json(output_dir / "h1_paired_by_run.json", h1)
    final_h1 = {
        **h1["pooled_question_run_pairs"],
        "all_runs_support_H1": h1["all_runs_support_H1"],
        "analysis_unit": "Paires question-run répétées entre seeds ; résultat descriptif.",
    }
    write_json(output_dir / "h1_paired_final.json", final_h1)
    agreement_path = output_dir / "manual_annotation_agreement.json"
    annotation_agreement = load_json(agreement_path) if agreement_path.exists() else None
    write_report_draft(
        output_dir / "report_final_draft.md", pipeline_summary, h1, annotation_agreement,
    )
    print(f"Finalisation écrite dans {output_dir}")


if __name__ == "__main__":
    main()
