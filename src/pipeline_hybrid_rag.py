"""
pipeline_hybrid_rag.py — Pipeline D : RAG hybride (Commun A+B, semaines 7-8)

Combine le retriever de schéma de A (SchemaRetrieverAdapter, stratégie
choisie lors de l'ablation préliminaire section 4.3) et la connaissance
métier de B (evidence BIRD, injectée telle quelle — section 3.2, jamais
de retrieval documentaire sur BIRD). C'est le pipeline utilisé pour
tester l'hypothèse H1 (comparaison avec B sur les erreurs structurelles
vs sémantiques, section 2.2 / 4.2).

Usage:
    python pipeline_hybrid_rag.py --questions ..\\data\\processed\\questions.json \
        --index_dir ..\\data\\index --output ..\\results\\pipelines_A_B_C_D\\pipeline_D\\generated_sql.json \
        --method hybrid --top_k 5 --limit 20
"""

import argparse
import json
from pathlib import Path

import sql_generator
from prompt_builder import PromptBuilder
from sql_generator import SQLGenerator
from retriever_adapters import SchemaRetrieverAdapter, CombinedRetriever


def run_pipeline_d(questions_path: str, index_dir: str, method: str, top_k: int,
                    model: str, temperature: float, limit: int | None,
                    run_index: int = 0, seed: int | None = None,
                    resume: bool = False,
                    checkpoint_path: str | None = None) -> list[dict]:
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if limit:
        questions = questions[:limit]

    results_by_question_id: dict[str, dict] = {}
    if resume:
        if checkpoint_path is None:
            raise ValueError("Un chemin de sortie est requis pour reprendre le pipeline D.")

        saved_path = Path(checkpoint_path)
        if saved_path.exists():
            with open(saved_path, "r", encoding="utf-8") as f:
                saved_results = json.load(f)
            if not isinstance(saved_results, list):
                raise ValueError(f"{saved_path} doit contenir une liste de résultats JSON.")

            expected_ids = {question["question_id"] for question in questions}
            for result in saved_results:
                if not isinstance(result, dict) or "question_id" not in result:
                    raise ValueError(f"Résultat de reprise invalide dans {saved_path}.")
                question_id = result["question_id"]
                if question_id not in expected_ids:
                    raise ValueError(
                        f"{saved_path} contient la question inconnue {question_id!r} "
                        "pour ce fichier --questions."
                    )
                if question_id in results_by_question_id:
                    raise ValueError(f"Question dupliquée dans le fichier de reprise : {question_id!r}.")
                results_by_question_id[question_id] = result
            print(f"Reprise : {len(results_by_question_id)} résultat(s) déjà disponible(s).")

    schema_adapter = SchemaRetrieverAdapter(index_dir, method=method, level="table")
    combined = CombinedRetriever(schema_adapter)

    # Même correctif que pour B/C : PromptBuilder() sans argument plante
    # dans sql_generator.py.
    sql_generator.PromptBuilder = lambda: PromptBuilder(combined)

    generator = SQLGenerator(model=model, temperature=temperature, top_k=top_k)

    for i, q in enumerate(questions, start=1):
        if q["question_id"] in results_by_question_id:
            print(f"[{i}/{len(questions)}] {q['question_id']} -> DÉJÀ PRÉSENT")
            continue

        combined.db_id = q["db_id"]
        combined.evidence = q.get("evidence", "")
        gen_result = generator.generate_sql(
            question=q["question"], top_k=top_k, run_index=run_index, seed=seed,
        )
        results_by_question_id[q["question_id"]] = {
            "question_id": q["question_id"],
            "db_id": q["db_id"],
            "sql_generated": gen_result.sql,
            "raw_response": gen_result.raw_response,
            "latency_seconds": gen_result.latency_seconds,
            "error": gen_result.error,
            "insufficient_context": gen_result.insufficient_context,
            "model": gen_result.model,
            "temperature": gen_result.temperature,
            "seed": gen_result.seed,
            "run_index": gen_result.run_index,
            "prompt_tokens": gen_result.prompt_tokens,
            "completion_tokens": gen_result.completion_tokens,
        }
        status = "OK" if gen_result.is_valid else (
            "CONTEXTE INSUFFISANT" if gen_result.insufficient_context else "ERREUR"
        )
        print(f"[{i}/{len(questions)}] {q['question_id']} -> {status}")

        if resume:
            save_results(
                [results_by_question_id[question["question_id"]] for question in questions
                 if question["question_id"] in results_by_question_id],
                checkpoint_path,
            )

    return [results_by_question_id[question["question_id"]] for question in questions]


def save_results(results: list[dict], output_path: str):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    temporary_path.replace(out_path)
    print(f"{len(results)} résultats écrits dans {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline D — génération SQL avec RAG hybride (schéma + métier).")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=["bm25", "embeddings", "hybrid"], default="hybrid",
                         help="Stratégie de retrieval de SCHÉMA retenue lors de l'ablation préliminaire (section 4.3)")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None, help="Seed du run reproductible")
    parser.add_argument("--run_index", type=int, default=0, help="Indice du run (0, 1, 2)")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de questions (test rapide)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reprendre depuis --output et sauvegarder après chaque question.",
    )
    args = parser.parse_args()

    results = run_pipeline_d(
        args.questions, args.index_dir, args.method, args.top_k,
        args.model, args.temperature, args.limit, run_index=args.run_index, seed=args.seed,
        resume=args.resume, checkpoint_path=args.output if args.resume else None,
    )
    save_results(results, args.output)


if __name__ == "__main__":
    main()
