"""
pipeline_baseline.py — Pipeline A : sans RAG (Étudiant A)

Génère du SQL pour chaque question de questions.json SANS aucun contexte
externe (ni schéma, ni connaissance métier) — le point de comparaison de
référence pour mesurer l'apport du retrieval (pipelines B/C/D).

Utilise SQLGenerator de B, mais avec builder = PromptBuilder(NullRetriever())
pour forcer un prompt sans contexte (voir retriever_adapters.py).

Usage:
    python pipeline_baseline.py --questions ..\\data\\processed\\questions.json \
        --output ..\\results\\pipelines_A_B_C_D\\pipeline_A\\generated_sql.json \
        --limit 20
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import sql_generator
from prompt_builder import PromptBuilder
from sql_generator import SQLGenerator
from retriever_adapters import NullRetriever


def run_pipeline_a(questions_path: str, model: str, temperature: float, limit: int | None) -> list[dict]:
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if limit:
        questions = questions[:limit]

    # Correctif : sql_generator.py fait `self.builder = PromptBuilder()` sans
    # argument dans son __init__, mais PromptBuilder exige un retriever ->
    # plante avant même qu'on puisse remplacer generator.builder après coup.
    # On patche donc la référence PromptBuilder vue par le module
    # sql_generator, le temps de construire le générateur, pour qu'elle
    # renvoie un prompt sans aucun contexte (baseline). A signaler à B.
    sql_generator.PromptBuilder = lambda: PromptBuilder(NullRetriever())
    generator = SQLGenerator(model=model, temperature=temperature)

    results = []
    for i, q in enumerate(questions, start=1):
        gen_result = generator.generate_sql(question=q["question"])
        results.append({
            "question_id": q["question_id"],
            "db_id": q["db_id"],
            "sql_generated": gen_result.sql,
            "raw_response": gen_result.raw_response,
            "latency_seconds": gen_result.latency_seconds,
            "error": gen_result.error,
        })
        print(f"[{i}/{len(questions)}] {q['question_id']} -> {'OK' if gen_result.is_valid else 'ERREUR'}")

    return results


def save_results(results: list[dict], output_path: str):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{len(results)} résultats écrits dans {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline A — génération SQL sans RAG (baseline).")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de questions (test rapide)")
    args = parser.parse_args()

    results = run_pipeline_a(args.questions, args.model, args.temperature, args.limit)
    save_results(results, args.output)


if __name__ == "__main__":
    main()