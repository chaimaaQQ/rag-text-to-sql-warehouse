"""
run_schema_ablation.py — Étude d'ablation retrieval, volet schéma (section 4.3)

Compare BM25 / embeddings / hybride sur le retrieval de schéma (niveau table
ET colonne), pour déterminer la stratégie "retrieval (fixé)" utilisée dans
l'étude principale (section 4.2). Réutilise evaluate_schema_retriever() de
evaluator.py, déjà implémenté — ce script se contente d'itérer sur toutes
les combinaisons et de sauvegarder un tableau récapitulatif.

Prérequis : data/evaluation/ground_truth.json doit déjà exister (sinon,
lancer d'abord `python evaluator.py ground-truth ...`, voir commande plus bas).

Usage:
    python run_schema_ablation.py --index_dir ..\\data\\index \
        --ground_truth ..\\data\\evaluation\\ground_truth.json \
        --questions ..\\data\\processed\\questions.json \
        --output ..\\results\\retrieval_ablation\\schema_ablation.json --k 5
"""

import argparse
import json
from pathlib import Path

from evaluator import evaluate_schema_retriever


METHODS = ["bm25", "embeddings", "hybrid"]
LEVELS = ["table", "column"]


def run_ablation(index_dir: str, ground_truth_path: str, questions_path: str, k: int) -> dict:
    results = {}
    for level in LEVELS:
        results[level] = {}
        for method in METHODS:
            print(f"--- level={level} method={method} ---")
            metrics = evaluate_schema_retriever(
                index_dir, level, ground_truth_path, questions_path, method, k
            )
            results[level][method] = metrics
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return results


def pick_best(results: dict, level: str = "table", metric: str = None) -> str:
    """Sélectionne la méthode avec le meilleur recall@k au niveau table
    (c'est cette valeur qui doit être reportée dans --method pour B/C/D)."""
    metric = metric or [k for k in results[level]["bm25"] if k.startswith("recall@")][0]
    return max(results[level], key=lambda m: results[level][m][metric] or 0)


def main():
    parser = argparse.ArgumentParser(description="Ablation retrieval de schéma (BM25/embeddings/hybride).")
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    results = run_ablation(args.index_dir, args.ground_truth, args.questions, args.k)

    best = pick_best(results, level="table")
    print(f"\n>>> Stratégie retenue (meilleur Recall@{args.k} niveau table) : {best}")
    print(">>> À utiliser comme --method dans pipeline_schema_rag.py (B) et pipeline_hybrid_rag.py (D)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_method_table_level": best}, f, ensure_ascii=False, indent=2)
    print(f"\nRésultats écrits dans {out_path}")


if __name__ == "__main__":
    main()