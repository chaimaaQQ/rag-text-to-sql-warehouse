"""
pipeline_schema_rag.py — Pipeline B : RAG schéma seul (Étudiant A)

Génère du SQL pour chaque question de questions.json en injectant UNIQUEMENT
le contexte de retriever_schema.py (tables/colonnes pertinentes) — pas de
connaissance métier (glossaire, KPI, règles). Sert à isoler la contribution
du schéma seul, indépendamment de C (métier seul) et D (hybride).

Usage:
    python pipeline_schema_rag.py --questions ..\\data\\processed\\questions.json \
        --index_dir ..\\data\\index --output ..\\results\\pipelines_A_B_C_D\\pipeline_B\\generated_sql.json \
        --method hybrid --top_k 5 --limit 20
"""

import argparse
import json
from pathlib import Path

from prompt_builder import PromptBuilder
from sql_generator import SQLGenerator
from retriever_adapters import SchemaRetrieverAdapter


def run_pipeline_b(questions_path: str, index_dir: str, method: str, top_k: int,
                    model: str, temperature: float, limit: int | None) -> list[dict]:
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if limit:
        questions = questions[:limit]

    adapter = SchemaRetrieverAdapter(index_dir, method=method, level="table")

    generator = SQLGenerator(model=model, temperature=temperature, top_k=top_k)
    generator.builder = PromptBuilder(adapter)  # correctif : voir note de coordination avec B

    results = []
    for i, q in enumerate(questions, start=1):
        adapter.db_id = q["db_id"]  # le schéma retrieval doit être restreint à la bonne base
        gen_result = generator.generate_sql(question=q["question"], top_k=top_k)
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
    parser = argparse.ArgumentParser(description="Pipeline B — génération SQL avec RAG schéma seul.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["bm25", "embeddings", "hybrid"], default="hybrid")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de questions (test rapide)")
    args = parser.parse_args()

    results = run_pipeline_b(
        args.questions, args.index_dir, args.method, args.top_k,
        args.model, args.temperature, args.limit,
    )
    save_results(results, args.output)


if __name__ == "__main__":
    main()
    