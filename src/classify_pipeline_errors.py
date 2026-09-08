"""
classify_pipeline_errors.py — Généralisation du script de classification
d'erreurs basé sur l'exécution réelle (SQLite), à tous les pipelines.

Contrairement à semantic_validator.py (heuristique lexicale, comparaison
SQL généré vs evidence), cette classification s'appuie sur les VRAIS
messages d'erreur retournés par SQLite lors de l'exécution — beaucoup
plus fiable et directement défendable dans le rapport.

Mapping utilisé pour le test de l'hypothèse H1 (section 2.2) :
    - Erreur STRUCTURELLE : unknown_column, unknown_table, ambiguous_column,
      syntax_error, unsupported_function, other_execution_error
      (tout ce qui empêche l'exécution de se dérouler correctement)
    - Erreur SÉMANTIQUE : wrong_execution_result
      (le SQL s'exécute sans erreur mais renvoie un résultat faux —
      signature d'une mauvaise règle métier / mauvais filtre / mauvaise
      formule de KPI)
    - Exclus (ni structurel ni sémantique) : insufficient_context, empty_sql

Usage (un pipeline à la fois, ou --all pour les 4) :
    python classify_pipeline_errors.py --pipeline D --results_dir ..\\results\\pipelines_A_B_C_D --output_dir ..\\results\\error_classification
    python classify_pipeline_errors.py --all --results_dir ..\\results\\pipelines_A_B_C_D --output_dir ..\\results\\error_classification
"""

import argparse
import json
from collections import Counter
from pathlib import Path


STRUCTURAL_CATEGORIES = {
    "unknown_column", "unknown_table", "ambiguous_column",
    "syntax_error", "unsupported_function", "other_execution_error",
}
SEMANTIC_CATEGORIES = {"wrong_execution_result"}
EXCLUDED_CATEGORIES = {"insufficient_context", "empty_sql"}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify(error, sql, insufficient_context) -> str:
    if insufficient_context:
        return "insufficient_context"
    if not sql or not sql.strip():
        return "empty_sql"
    if not error:
        return "wrong_execution_result"
    error_lower = error.lower()
    if "no such column" in error_lower:
        return "unknown_column"
    if "no such table" in error_lower:
        return "unknown_table"
    if "no such function" in error_lower:
        return "unsupported_function"
    if "syntax error" in error_lower:
        return "syntax_error"
    if "ambiguous" in error_lower:
        return "ambiguous_column"
    if "sql vide" in error_lower:
        return "empty_sql"
    return "other_execution_error"


def classify_generated_execution(generated_path: Path, execution_path: Path) -> list[dict]:
    """Classify one generated SQL file and its paired execution report.

    This path-level API is used by the three-run main-study orchestrator;
    keeping it here guarantees exactly the same category definitions for the
    old single-run layout and the new reproducible layout.
    """
    execution = load_json(execution_path)
    generated = load_json(generated_path)
    generated_by_id = {item["question_id"]: item for item in generated}

    per_question = []
    for item in execution["details"]:
        qid = item["question_id"]
        match = item.get("execution_match")
        gen_item = generated_by_id.get(qid, {})
        sql = gen_item.get("sql_generated", "")
        insufficient_context = gen_item.get("insufficient_context", False)

        if match:
            category = "match"
        else:
            category = classify(item.get("error"), sql, insufficient_context)

        per_question.append({
            "question_id": qid,
            "db_id": item.get("db_id"),
            "category": category,
            "execution_match": bool(match) if match is not None else None,
        })

    return per_question


def classify_pipeline(pipeline: str, results_dir: Path) -> list[dict]:
    base = results_dir / f"pipeline_{pipeline}"
    return classify_generated_execution(
        base / "generated_sql.json", base / "execution_accuracy.json",
    )


def print_summary(pipeline: str, per_question: list[dict]):
    counter = Counter(r["category"] for r in per_question)
    total = len(per_question)

    print("=" * 80)
    print(f"PIPELINE {pipeline} — CLASSIFICATION DES ERREURS (basée sur l'exécution)")
    print("=" * 80)
    for category, count in counter.most_common():
        print(f"{category:25s} {count:4d} ({100 * count / total:6.2f}%)")

    n_excluded = sum(counter[c] for c in EXCLUDED_CATEGORIES)
    n_attempted = total - n_excluded
    n_struct = sum(counter[c] for c in STRUCTURAL_CATEGORIES)
    n_sem = sum(counter[c] for c in SEMANTIC_CATEGORIES)
    n_match = counter["match"]

    print()
    print(f"Tentatives réelles (hors refus/vide) : {n_attempted}")
    if n_attempted:
        print(f"  -> match (résultat correct)      : {n_match} ({100*n_match/n_attempted:.1f}%)")
        print(f"  -> erreur structurelle            : {n_struct} ({100*n_struct/n_attempted:.1f}%)")
        print(f"  -> erreur sémantique               : {n_sem} ({100*n_sem/n_attempted:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Classification d'erreurs basée sur l'exécution SQLite réelle.")
    parser.add_argument("--pipeline", choices=["A", "B", "C", "D"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if not args.pipeline and not args.all:
        parser.error("Spécifiez --pipeline X ou --all")

    pipelines = ["A", "B", "C", "D"] if args.all else [args.pipeline]
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pipeline in pipelines:
        per_question = classify_pipeline(pipeline, results_dir)
        print_summary(pipeline, per_question)

        out_path = output_dir / f"execution_based_{pipeline}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(per_question, f, ensure_ascii=False, indent=2)
        print(f"Écrit dans {out_path}\n")


if __name__ == "__main__":
    main()
