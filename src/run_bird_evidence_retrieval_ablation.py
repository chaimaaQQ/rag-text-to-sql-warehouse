"""
run_bird_evidence_retrieval_ablation.py — Test de retrieval documentaire
masqué sur BIRD-Evidence-Corpus (section 3.2, terrain secondaire).

Rappel du protocole imposé par le cahier des charges : pour CHAQUE
question BIRD, on masque son evidence d'origine et on teste si le
système la retrouve parmi les distracteurs (les evidence des ~500 autres
questions). C'est le SEUL contexte où Knowledge Recall@k a un sens sur
BIRD — jamais sur le retrieval de schéma (terrain de l'étudiant A) ni
sur les pipelines B/C/D eux-mêmes (qui reçoivent evidence tel quel,
sans retrieval — section 3.2).

"Masquer" ici signifie : la requête est la QUESTION (pas l'evidence),
et on vérifie si le document evidence correspondant à cette question
précise ressort dans le top-k parmi les 500 candidats (aucun document
n'est retiré du corpus — le masquage porte sur le fait de ne pas donner
l'evidence en entrée, pas sur l'exclusion physique du corpus).

Usage:
    python run_bird_evidence_retrieval_ablation.py \
        --questions ..\\data\\processed\\questions.json \
        --knowledge_base ..\\data\\knowledge_base\\knowledge_base.json \
        --output ..\\results\\retrieval_ablation\\bird_evidence_ablation.json \
        --k 5 --sample_size 150
"""

import argparse
import json
import random
from pathlib import Path

from business_retrieval_utils import build_index, retrieve, recall_at_k, precision_at_k


METHODS = ["bm25", "embeddings", "hybrid"]


def evaluate_method(questions: list[dict], bird_id_to_doc_id: dict, docs: list[dict],
                     bm25, embeddings, model_name, method: str, k: int) -> dict:
    recalls, precisions, mrrs = [], [], []
    n_skipped = 0

    for q in questions:
        gold_doc_id = bird_id_to_doc_id.get(_extract_bird_id(q))
        if not gold_doc_id or not q.get("evidence"):
            n_skipped += 1
            continue

        gold_ids = {gold_doc_id}
        # La requête est la QUESTION seule (l'evidence est "masquée", jamais
        # donnée en entrée du retriever).
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

    n = len(recalls)
    return {
        f"recall@{k}": (sum(recalls) / n) if n else None,
        f"precision@{k}": (sum(precisions) / n) if n else None,
        "mrr": (sum(mrrs) / n) if n else None,
        "n_questions": n,
        "n_skipped": n_skipped,
    }


def _extract_bird_id(q: dict) -> int | None:
    qid = q["question_id"]  # ex: "bird_mini_1471"
    try:
        return int(qid.rsplit("_", 1)[-1])
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Retrieval masqué sur BIRD-Evidence-Corpus (section 3.2).")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--knowledge_base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=150,
                         help="Nombre de questions pour l'ablation (comparaison courte, section 4.3). 0 = toutes.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(args.knowledge_base, "r", encoding="utf-8") as f:
        all_docs = json.load(f)

    docs = [d for d in all_docs if d["type"] == "bird_evidence"]
    bird_id_to_doc_id = {d["metadata"]["bird_id"]: d["id"] for d in docs}
    print(f"Corpus BIRD-Evidence-Corpus : {len(docs)} documents")

    if args.sample_size:
        random.seed(args.seed)
        questions = random.sample(questions, min(args.sample_size, len(questions)))
    print(f"Questions de test (échantillon) : {len(questions)}")

    bm25, embeddings, model_name = build_index(docs)

    results = {}
    for method in METHODS:
        print(f"\n--- method={method} ---")
        metrics = evaluate_method(questions, bird_id_to_doc_id, docs, bm25, embeddings, model_name, method, args.k)
        results[method] = metrics
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    best = max(results, key=lambda m: results[m][f"recall@{args.k}"] or 0)
    print(f"\n>>> Stratégie retenue (Knowledge Recall@{args.k}) : {best}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_method": best, "corpus": "bird_evidence_corpus", "n_docs": len(docs)},
                   f, ensure_ascii=False, indent=2)
    print(f"Résultats écrits dans {out_path}")


if __name__ == "__main__":
    main()