"""
build_tpcds_eval_questions.py — Construction du jeu de questions de test
pour le corpus documentaire TPC-DS (Module 3, section 3.3).

Génère, à partir de kpis.json et business_rules.json, des questions en
langage naturel dont la réponse nécessite CE document précis (label de
pertinence). C'est le "construit par l'étudiant B" que la section 3.3
demande de faire valider ensuite par l'étudiant A (cross_validate_business_corpus.py).

Les questions sont des GABARITS DE DÉPART, pas des questions déjà validées :
relisez-les et reformulez celles qui sonnent trop mécaniquement avant la
validation croisée — le but est d'avoir des questions réalistes, pas de
tricher en collant juste le texte source du document dedans (ce qui
rendrait le retrieval artificiellement facile, cf. le même problème que
BIRD identifié section 3.1).

Usage:
    python build_tpcds_eval_questions.py \
        --kpis ..\\data\\business_docs\\kpis.json \
        --business_rules ..\\data\\business_docs\\business_rules.json \
        --knowledge_base ..\\data\\knowledge_base\\knowledge_base.json \
        --output ..\\data\\evaluation\\tpcds_eval_questions.json
"""

import argparse
import json
import random
from pathlib import Path


KPI_TEMPLATES = [
    "How is {name} calculated?",
    "What is the formula for {name}?",
    "How do we compute the {name_lower} for the {table} table?",
    "Which measure should I aggregate to get {name_lower}?",
]

RULE_TEMPLATES = [
    "What is the rule for calculating {title_lower}?",
    "Which column should be used to compute {title_lower}?",
    "What business rule applies to {title_lower}?",
    "How should {title_lower} be computed according to company policy?",
]


def build_kb_lookup(knowledge_base: list[dict]) -> dict:
    """Map (type, title) -> id, pour retrouver l'id KB0xxx correspondant
    à un KPI ou une règle métier."""
    lookup = {}
    for doc in knowledge_base:
        lookup[(doc["type"], doc["title"])] = doc["id"]
    return lookup


def generate_questions(kpis: list[dict], rules: list[dict], kb_lookup: dict, seed: int) -> list[dict]:
    random.seed(seed)
    questions = []
    qid = 1

    for kpi in kpis:
        doc_id = kb_lookup.get(("kpi", kpi["name"]))
        if not doc_id:
            continue
        template = random.choice(KPI_TEMPLATES)
        question = template.format(name=kpi["name"], name_lower=kpi["name"].lower(), table=kpi["table"])
        questions.append({
            "question_id": f"tpcds_eval_{qid:03d}",
            "question": question,
            "relevant_doc_ids": [doc_id],
            "source_type": "kpi",
            "source_id": kpi["id"],
        })
        qid += 1

    for rule in rules:
        doc_id = kb_lookup.get(("business_rule", rule["title"]))
        if not doc_id:
            continue
        template = random.choice(RULE_TEMPLATES)
        question = template.format(title_lower=rule["title"].lower())
        # Une règle métier référence souvent aussi le(s) KPI(s) associé(s) :
        # les documents KPI liés sont donc AUSSI pertinents pour répondre.
        related_ids = [doc_id]
        for kpi_name in rule.get("related_kpis", []):
            related_doc_id = kb_lookup.get(("kpi", kpi_name))
            if related_doc_id:
                related_ids.append(related_doc_id)
        questions.append({
            "question_id": f"tpcds_eval_{qid:03d}",
            "question": question,
            "relevant_doc_ids": related_ids,
            "source_type": "business_rule",
            "source_id": rule["id"],
        })
        qid += 1

    return questions


def main():
    parser = argparse.ArgumentParser(description="Génère les questions de test du corpus TPC-DS (section 3.3).")
    parser.add_argument("--kpis", required=True)
    parser.add_argument("--business_rules", required=True)
    parser.add_argument("--knowledge_base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.kpis, "r", encoding="utf-8") as f:
        kpis = json.load(f)
    with open(args.business_rules, "r", encoding="utf-8") as f:
        rules = json.load(f)
    with open(args.knowledge_base, "r", encoding="utf-8") as f:
        kb = json.load(f)

    kb_lookup = build_kb_lookup(kb)
    questions = generate_questions(kpis, rules, kb_lookup, args.seed)

    print(f"{len(questions)} questions générées ({sum(1 for q in questions if q['source_type']=='kpi')} KPI, "
          f"{sum(1 for q in questions if q['source_type']=='business_rule')} règles métier)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"Écrit dans {out_path}")
    print("\nIMPORTANT : relisez et reformulez les questions avant la validation croisée "
          "(section 3.3) — ce sont des gabarits de départ, pas un jeu figé.")


if __name__ == "__main__":
    main()