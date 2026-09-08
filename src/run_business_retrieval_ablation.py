"""
run_business_retrieval_ablation.py — Ablation retrieval documentaire,
terrain TPC-DS (section 3.2 / 4.3, volet documentaire de l'étudiant B).

Compare BM25 / embeddings / hybride sur le retrieval documentaire métier :
corpus = glossary + kpi + business_rule (51 documents, PAS bird_evidence —
cf. section 3.2, ce terrain est distinct de BIRD-Evidence-Corpus).

Prérequis : avoir lancé build_tpcds_eval_questions.py au préalable.

Usage:
    python run_business_retrieval_ablation.py \
        --questions ..\\data\\evaluation\\tpcds_eval_questions.json \
        --knowledge_base ..\\data\\knowledge_base\\knowledge_base.json \
        --output ..\\results\\retrieval_ablation\\business_ablation.json --k 5
"""

import argparse
import json
from pathlib import Path

from business_retrieval_utils import build_index, retrieve, recall_at_k, precision_at_k


METHODS = ["bm25", "embeddings", "hybrid"]
CORPUS_TYPES = {"glossary", "kpi", "business_rule"}


def evaluate_method(questions: list[dict], docs: list[dict], bm25, embeddings, model_name,
                     method: str, k: int) -> dict:
    recalls, precisions, mrrs = [], [], []

    for q in questions:
        gold_ids = set(q["relevant_doc_ids"])
        results = retrieve(q["question"], docs, bm25, embeddings, model_name, k=k, method=method)
        retrieved_ids = [r["doc_id"] for r in results]

        recalls.append(recall_at_k(retrieved_ids, gold_ids))
        precisions.append(precision_at_k(retrieved_ids, gold_ids, k))

        rr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in gold_ids:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    n = len(questions)
    return {
        f"recall@{k}": sum(recalls) / n,
        f"precision@{k}": sum(precisions) / n,
        "mrr": sum(mrrs) / n,
        "n_questions": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation retrieval documentaire (TPC-DS, section 4.3).")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--knowledge_base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(args.knowledge_base, "r", encoding="utf-8") as f:
        all_docs = json.load(f)

    docs = [d for d in all_docs if d["type"] in CORPUS_TYPES]
    print(f"Corpus TPC-DS : {len(docs)} documents ({', '.join(sorted(CORPUS_TYPES))})")
    print(f"Questions de test : {len(questions)}")

    bm25, embeddings, model_name = build_index(docs)

    results = {}
    for method in METHODS:
        print(f"\n--- method={method} ---")
        metrics = evaluate_method(questions, docs, bm25, embeddings, model_name, method, args.k)
        results[method] = metrics
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    best = max(results, key=lambda m: results[m][f"recall@{args.k}"])
    print(f"\n>>> Stratégie retenue (Knowledge Recall@{args.k}) : {best}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_method": best, "corpus": "tpcds", "n_docs": len(docs)},
                   f, ensure_ascii=False, indent=2)
    print(f"Résultats écrits dans {out_path}")


if __name__ == "__main__":
    main()