"""
cross_validate_business_corpus.py — Validation croisée A/B du corpus
métier TPC-DS (section 3.3, livrable explicite de la semaine 3).

DEUX ÉTAPES :

1) --mode sample : tire un échantillon (>=30 par défaut) de paires
   question/document parmi tpcds_eval_questions.json et écrit une feuille
   CSV à remplir À LA MAIN par l'étudiant A (qui n'a pas construit ces
   questions), avec une colonne "agree" (oui/non) et "comment".

2) --mode score : une fois la feuille remplie par l'étudiant A, relit le
   CSV et calcule le taux d'accord + liste les désaccords, à documenter
   tels quels dans le rapport (jamais résolus silencieusement, section 3.3).

Usage:
    # Étape 1 (étudiant B, ou vous) : générer la feuille à remplir
    python cross_validate_business_corpus.py --mode sample \
        --questions ..\\data\\evaluation\\tpcds_eval_questions.json \
        --knowledge_base ..\\data\\knowledge_base\\knowledge_base.json \
        --output ..\\results\\cross_validation\\tpcds_sheet.csv --n 30

    # (remplir la colonne "agree" à la main : oui / non)

    # Étape 2 (une fois rempli) : calculer le taux d'accord
    python cross_validate_business_corpus.py --mode score \
        --sheet ..\\results\\cross_validation\\tpcds_sheet.csv \
        --output ..\\results\\cross_validation\\tpcds_agreement_report.json
"""

import argparse
import csv
import json
import random
from pathlib import Path


def sample_sheet(questions_path: str, kb_path: str, output_path: str, n: int, seed: int):
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = {d["id"]: d for d in json.load(f)}

    random.seed(seed)
    n = min(n, len(questions))
    sample = random.sample(questions, n)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question", "relevant_doc_id", "relevant_doc_title",
                          "relevant_doc_content", "agree(oui/non)", "comment"])
        for q in sample:
            for doc_id in q["relevant_doc_ids"]:
                doc = kb.get(doc_id, {})
                writer.writerow([
                    q["question_id"], q["question"], doc_id,
                    doc.get("title", ""), doc.get("content", ""), "", "",
                ])

    n_rows = sum(len(q["relevant_doc_ids"]) for q in sample)
    print(f"{n} questions échantillonnées ({n_rows} lignes question/document au total, "
          f"une question peut avoir plusieurs documents pertinents).")
    print(f"Feuille écrite dans {out_path}")
    print("\n>>> À FAIRE : l'étudiant A remplit la colonne 'agree(oui/non)' (et 'comment' si désaccord),")
    print(">>> puis relancez ce script avec --mode score sur le fichier rempli.")


def score_sheet(sheet_path: str, output_path: str):
    with open(sheet_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    filled = [r for r in rows if r["agree(oui/non)"].strip()]
    if not filled:
        print("ATTENTION : aucune ligne n'a la colonne 'agree(oui/non)' remplie. "
              "Remplissez la feuille avant de calculer le score.")
        return

    n_total = len(filled)
    n_agree = sum(1 for r in filled if r["agree(oui/non)"].strip().lower() in ("oui", "yes", "y", "o"))
    disagreements = [r for r in filled if r["agree(oui/non)"].strip().lower() not in ("oui", "yes", "y", "o")]

    agreement_rate = n_agree / n_total

    print(f"Lignes évaluées : {n_total}")
    print(f"Accords : {n_agree} ({100*agreement_rate:.1f}%)")
    print(f"Désaccords : {len(disagreements)}")
    if disagreements:
        print("\nDétail des désaccords (à documenter dans le rapport, section 3.3) :")
        for r in disagreements:
            print(f"  - {r['question_id']} / {r['relevant_doc_id']} : {r['comment'] or '(pas de commentaire)'}")

    report = {
        "n_total": n_total,
        "n_agree": n_agree,
        "agreement_rate": agreement_rate,
        "disagreements": [
            {"question_id": r["question_id"], "doc_id": r["relevant_doc_id"],
             "question": r["question"], "comment": r["comment"]}
            for r in disagreements
        ],
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nRapport écrit dans {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Validation croisée A/B du corpus métier (section 3.3).")
    parser.add_argument("--mode", choices=["sample", "score"], required=True)
    parser.add_argument("--questions")
    parser.add_argument("--knowledge_base")
    parser.add_argument("--sheet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "sample":
        if not args.questions or not args.knowledge_base:
            parser.error("--mode sample nécessite --questions et --knowledge_base")
        sample_sheet(args.questions, args.knowledge_base, args.output, args.n, args.seed)
    else:
        if not args.sheet:
            parser.error("--mode score nécessite --sheet")
        score_sheet(args.sheet, args.output)


if __name__ == "__main__":
    main()