"""
apply_cross_validation.py — Applique les décisions de la validation
croisée (section 3.3) pour nettoyer le ground truth avant l'ablation
retrieval documentaire (section 4.3).

Retire des relevant_doc_ids toute paire question/document marquée "non"
dans la feuille de validation croisée. Les questions qui se retrouvent
sans aucun document pertinent restant sont retirées du jeu d'évaluation
(et listées, pour transparence).

Usage:
    python apply_cross_validation.py \
        --questions ..\\data\\evaluation\\tpcds_eval_questions.json \
        --sheet ..\\results\\cross_validation\\tpcds_sheet.csv \
        --output ..\\data\\evaluation\\tpcds_eval_questions_validated.json
"""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Nettoie le ground truth TPC-DS selon la validation croisée.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = {q["question_id"]: q for q in json.load(f)}

    with open(args.sheet, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rejected = {}  # question_id -> set(doc_id) marqués "non"
    for r in rows:
        agree = r["agree(oui/non)"].strip().lower()
        if agree in ("non", "no", "n"):
            rejected.setdefault(r["question_id"], set()).add(r["relevant_doc_id"])

    n_cleaned = 0
    n_dropped_questions = 0
    validated = []
    for qid, q in questions.items():
        to_remove = rejected.get(qid, set())
        kept = [d for d in q["relevant_doc_ids"] if d not in to_remove]
        if to_remove:
            n_cleaned += 1
        if not kept:
            n_dropped_questions += 1
            print(f"ATTENTION : {qid} n'a plus aucun document pertinent après validation -> retirée du jeu.")
            continue
        new_q = dict(q)
        new_q["relevant_doc_ids"] = kept
        validated.append(new_q)

    print(f"{len(validated)}/{len(questions)} questions conservées "
          f"({n_cleaned} nettoyées, {n_dropped_questions} retirées).")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validated, f, ensure_ascii=False, indent=2)
    print(f"Écrit dans {out_path}")


if __name__ == "__main__":
    main()