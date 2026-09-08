"""
semantic_validator.py — Module 6, volet sémantique (Étudiant B)

Classifie les erreurs SEMANTIQUES (mauvaise règle de calcul, mauvaise
définition de filtre métier, mauvaise formule de KPI) d'un lot de requêtes
générées, EN COMPLEMENT de sql_validator.py qui classe les erreurs
STRUCTURELLES (table/colonne inexistante, jointure incorrecte).

IMPORTANT (section 8 du cahier des charges) : cette classification est
heuristique (comparaison lexicale SQL généré vs `evidence` + `sql_gold`).
Elle doit être validée manuellement sur un échantillon avant d'être utilisée
dans le test de l'hypothèse H1 — c'est un point de départ, pas un oracle.
Le rapport final doit documenter le taux d'accord inter-annotateur si vous
la corrigez à la main (comme pour la validation croisée A/B, section 3.3).

Heuristique utilisée :
  1. On ignore les questions dont le SQL est structurellement invalide
     (déjà classées par sql_validator.py — pas la peine de les compter deux
     fois comme "sémantiques").
  2. On extrait de `evidence` les valeurs discriminantes (nombres, chaînes
     entre quotes, comparateurs) — ce sont les éléments d'une règle métier
     ou d'un filtre qui DOIVENT apparaître dans une requête correcte.
  3. Si le SQL structurellement valide ne contient AUCUNE de ces valeurs
     alors qu'evidence en contient, on le marque "semantic_error" avec le
     détail des éléments manquants.
  4. On compare aussi l'agrégation utilisée (SUM/COUNT/AVG/...) entre le
     SQL généré et sql_gold : une divergence d'agrégation sur la même
     colonne est un signal fort de mauvaise formule de KPI.

Usage:
    python semantic_validator.py --generated ..\\results\\pipelines_A_B_C_D\\pipeline_B\\generated_sql.json \
        --structural ..\\data\\evaluation\\structural_A.json \
        --questions ..\\data\\processed\\questions.json \
        --output ..\\results\\error_classification\\semantic_B.json
"""

import argparse
import json
import re
from pathlib import Path


AGG_FUNCS = ["SUM", "COUNT", "AVG", "MIN", "MAX"]


def extract_discriminants(evidence: str) -> list[str]:
    """Extrait les valeurs qu'une requête correcte doit vraisemblablement
    reprendre : chaînes entre quotes ('EUR', 'CZK', ...) et nombres."""
    if not evidence:
        return []
    quoted = re.findall(r"'([^']+)'", evidence)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", evidence)
    return list(dict.fromkeys(quoted + numbers))  # dédoublonne en gardant l'ordre


def extract_agg(sql: str) -> set[str]:
    if not sql:
        return set()
    return {f for f in AGG_FUNCS if re.search(rf"\b{f}\s*\(", sql, re.IGNORECASE)}


def classify_semantic(sql_generated: str, sql_gold: str, evidence: str) -> dict:
    discriminants = extract_discriminants(evidence)
    sql_lower = (sql_generated or "").lower()

    missing = [d for d in discriminants if d.lower() not in sql_lower]

    agg_generated = extract_agg(sql_generated)
    agg_gold = extract_agg(sql_gold)
    agg_mismatch = bool(agg_gold) and agg_generated != agg_gold

    is_semantic_error = bool(missing) or agg_mismatch

    error_detail = []
    if missing:
        error_detail.append(f"Élément(s) métier de l'evidence absent(s) du SQL : {missing}")
    if agg_mismatch:
        error_detail.append(f"Agrégation divergente : généré={sorted(agg_generated)} attendu={sorted(agg_gold)}")

    return {
        "is_semantic_error": is_semantic_error,
        "error_type": "semantic_error" if is_semantic_error else None,
        "error_detail": " | ".join(error_detail) if error_detail else None,
        "missing_evidence_terms": missing,
        "agg_generated": sorted(agg_generated),
        "agg_gold": sorted(agg_gold),
    }


def run(generated_path: str, structural_path: str, questions_path: str) -> list[dict]:
    with open(generated_path, "r", encoding="utf-8") as f:
        generated = {item["question_id"]: item for item in json.load(f)}
    with open(structural_path, "r", encoding="utf-8") as f:
        structural = {item["question_id"]: item for item in json.load(f)}
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = {q["question_id"]: q for q in json.load(f)}

    results = []
    for qid, gen in generated.items():
        struct = structural.get(qid, {})
        q = questions.get(qid, {})

        entry = {
            "question_id": qid,
            "db_id": gen.get("db_id"),
            "structurally_valid": struct.get("is_valid", None),
        }

        if struct.get("is_valid") is False:
            # Ne pas double-compter : déjà classé comme erreur structurelle.
            entry["is_semantic_error"] = None
            entry["error_type"] = None
            entry["error_detail"] = "Ignoré : requête déjà structurellement invalide."
        else:
            semantic = classify_semantic(
                gen.get("sql_generated", ""), q.get("sql_gold", ""), q.get("evidence", "")
            )
            entry.update(semantic)

        results.append(entry)

    n_semantic = sum(1 for r in results if r.get("is_semantic_error"))
    n_evaluated = sum(1 for r in results if r.get("is_semantic_error") is not None)
    if n_evaluated:
        print(f"{n_semantic}/{n_evaluated} requêtes (structurellement valides) classées en erreur sémantique "
              f"({100 * n_semantic / n_evaluated:.1f}%)")

    return results


def save_results(results: list[dict], output_path: str):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Rapport écrit dans {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Classification heuristique des erreurs sémantiques.")
    parser.add_argument("--generated", required=True, help="results/pipelines_A_B_C_D/pipeline_X/generated_sql.json")
    parser.add_argument("--structural", required=True, help="data/evaluation/structural_X.json (sql_validator.py)")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = run(args.generated, args.structural, args.questions)
    save_results(results, args.output)


if __name__ == "__main__":
    main()