"""
evaluator.py — Commun (A + B)

Deux responsabilités :
1. Extraction du ground truth : parse chaque sql_gold pour en extraire les
   tables/colonnes réellement utilisées (spécifique au track schéma, A).
2. Métriques de retrieval génériques : Recall@k, Precision@k, MRR, nDCG@k.
   Ces fonctions sont volontairement génériques (listes d'identifiants) pour
   être réutilisables aussi bien par A (retriever_schema.py, tables/colonnes)
   que par B (retriever_docs.py, chunks de documents métier).

Dépendance :
    pip install sqlglot

Usage (extraction du ground truth, track A) :
    python evaluator.py ground-truth --questions data/processed/questions.json \
        --schemas_dir data/schemas --output data/evaluation/ground_truth.json

Usage (évaluation d'un retriever déjà construit, track A) :
    python evaluator.py eval-schema --index_dir data/index --level table \
        --ground_truth data/evaluation/ground_truth.json --method hybrid --k 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import sqlglot
from sqlglot import exp




def load_schema_lookup(schemas_dir: str) -> dict:
    """Construit {db_id: {table_name_lower: [colonnes...]}} pour désambiguïser
    les colonnes sans préfixe de table explicite dans le SQL."""
    lookup = {}
    for path in Path(schemas_dir).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        lookup[schema["db_id"]] = {
            table["name"].lower(): [c["name"] for c in table["columns"]]
            for table in schema["tables"]
        }
    return lookup


def extract_tables_and_columns(sql: str, db_id: str, schema_lookup: dict) -> dict:
    """Parse une requête SQL (dialecte sqlite) et retourne les tables/colonnes
    réellement utilisées. Gère les alias et les jointures via sqlglot."""
    tables_found, columns_found, parse_error = set(), set(), False

    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")

        alias_to_table = {}
        for table_node in parsed.find_all(exp.Table):
            table_name = table_node.name.lower()
            tables_found.add(table_name)
            if table_node.alias:
                alias_to_table[table_node.alias.lower()] = table_name

        db_tables = schema_lookup.get(db_id, {})
        for col_node in parsed.find_all(exp.Column):
            col_name = col_node.name
            table_ref = col_node.table.lower() if col_node.table else None
            resolved_table = alias_to_table.get(table_ref, table_ref)

            if resolved_table is None and len(tables_found) == 1:
                resolved_table = next(iter(tables_found))

            if resolved_table:
                columns_found.add(f"{resolved_table}.{col_name}")
            else:
                candidates = [t for t, cols in db_tables.items() if col_name in cols]
                if len(candidates) == 1:
                    columns_found.add(f"{candidates[0]}.{col_name}")

    except Exception:
        parse_error = True

    return {"tables": sorted(tables_found), "columns": sorted(columns_found), "parse_error": parse_error}


def build_ground_truth(questions_path: str, schemas_dir: str) -> list[dict]:
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    schema_lookup = load_schema_lookup(schemas_dir)
    ground_truth, n_errors = [], 0

    for q in questions:
        result = extract_tables_and_columns(q["sql_gold"], q["db_id"], schema_lookup)
        n_errors += result["parse_error"]
        ground_truth.append({
            "question_id": q["question_id"],
            "db_id": q["db_id"],
            "tables_gold": result["tables"],
            "columns_gold": result["columns"],
            "parse_error": result["parse_error"],
        })

    if n_errors:
        print(f"Attention : {n_errors} requête(s) sur {len(questions)} n'ont pas pu être parsées.")
    return ground_truth




def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return None
    top_k = set(retrieved[:k])
    return len(top_k & set(gold)) / len(gold)


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return None
    return len(set(top_k) & set(gold)) / len(top_k)


def mrr(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    top_k = retrieved[:k]

    dcg = sum(1.0 / np.log2(rank + 1) for rank, doc_id in enumerate(top_k, start=1) if doc_id in gold_set)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_run(all_retrieved: list[list[str]], all_gold: list[list[str]], k: int) -> dict:
    """Moyenne les métriques sur un ensemble de questions.
    all_retrieved[i] = liste ordonnée des doc_id retournés pour la question i.
    all_gold[i] = liste des doc_id réellement pertinents pour la question i.
    """
    recalls, precisions, mrrs, ndcgs = [], [], [], []

    for retrieved, gold in zip(all_retrieved, all_gold):
        if not gold:
            continue
        recalls.append(recall_at_k(retrieved, gold, k))
        precisions.append(precision_at_k(retrieved, gold, k))
        mrrs.append(mrr(retrieved, gold))
        ndcgs.append(ndcg_at_k(retrieved, gold, k))

    return {
        f"recall@{k}": float(np.mean(recalls)) if recalls else None,
        f"precision@{k}": float(np.mean(precisions)) if precisions else None,
        "mrr": float(np.mean(mrrs)) if mrrs else None,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else None,
        "n_questions": len(recalls),
    }




def evaluate_schema_retriever(index_dir: str, level: str, ground_truth_path: str,
                               questions_path: str, method: str, k: int) -> dict:
    import retriever_schema as rs

    docs, bm25, embeddings, model_name = rs.load_index(index_dir, level)

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = {q["question_id"]: q for q in json.load(f)}

    gold_key = "tables_gold" if level == "table" else "columns_gold"

    all_retrieved, all_gold = [], []
    for gt in ground_truth:
        if gt["parse_error"] or not gt[gold_key]:
            continue
        question_text = questions[gt["question_id"]]["question"]
        results = rs.retrieve(question_text, docs, bm25, embeddings, model_name,
                               db_id=gt["db_id"], k=k, method=method)

        if level == "table":
            retrieved_ids = [r["doc"]["table_name"].lower() for r in results]
        else:
            retrieved_ids = [f"{r['doc']['table_name'].lower()}.{r['doc']['column_name']}" for r in results]

        all_retrieved.append(retrieved_ids)
        all_gold.append(gt[gold_key])

    return evaluate_run(all_retrieved, all_gold, k)




def main():
    parser = argparse.ArgumentParser(description="Extraction du ground truth et évaluation du retrieval.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gt_p = subparsers.add_parser("ground-truth", help="Extrait tables/colonnes gold depuis sql_gold")
    gt_p.add_argument("--questions", required=True)
    gt_p.add_argument("--schemas_dir", required=True)
    gt_p.add_argument("--output", required=True)

    eval_p = subparsers.add_parser("eval-schema", help="Évalue le retriever de schéma (Recall@k, Precision@k, MRR, nDCG@k)")
    eval_p.add_argument("--index_dir", required=True)
    eval_p.add_argument("--level", choices=["table", "column"], default="table")
    eval_p.add_argument("--ground_truth", required=True)
    eval_p.add_argument("--questions", required=True)
    eval_p.add_argument("--method", choices=["bm25", "embeddings", "hybrid"], default="hybrid")
    eval_p.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "ground-truth":
        ground_truth = build_ground_truth(args.questions, args.schemas_dir)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, ensure_ascii=False, indent=2)
        print(f"{len(ground_truth)} entrées écrites dans {out_path}")

    elif args.command == "eval-schema":
        metrics = evaluate_schema_retriever(
            args.index_dir, args.level, args.ground_truth, args.questions, args.method, args.k
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()