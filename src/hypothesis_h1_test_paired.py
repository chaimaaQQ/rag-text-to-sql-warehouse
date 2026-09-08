"""
hypothesis_h1_test_paired.py — Test H1, comparaison APPARIÉE (matched)

PROBLÈME identifié sur hypothesis_h1_test_v2.py : B et D n'ont pas le même
taux de refus (B refuse 70,9% des questions, D seulement 22,7%). La
comparaison "taux d'erreur parmi les tentatives" de chaque pipeline
compare donc deux POPULATIONS DE QUESTIONS DIFFÉRENTES : B ne tente que
les questions "faciles" (celles où le schéma seul suffit à le rassurer),
alors que D tente aussi les questions plus dures que B a fuies. Une
différence structurelle observée peut donc être un artefact de cette
différence de composition, pas une vraie différence de qualité du
retrieval de schéma (qui est pourtant identique entre B et D).

CORRECTIF : restreindre la comparaison à l'INTERSECTION des question_id
où B ET D ont tous les deux réellement tenté une réponse (catégorie
'match' ou une erreur d'exécution — PAS 'insufficient_context' ni
'empty_sql'). Sur ce sous-ensemble apparié, B et D sont comparés sur
EXACTEMENT les mêmes questions — élimine le biais de sélection.

Usage:
    python hypothesis_h1_test_paired.py \
        --error_classification_dir ..\\results\\error_classification \
        --output ..\\results\\hypothesis_H1\\h1_test_result_paired.json
"""

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest

from classify_pipeline_errors import STRUCTURAL_CATEGORIES, SEMANTIC_CATEGORIES, EXCLUDED_CATEGORIES


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return {r["question_id"]: r for r in json.load(f)}


def run_mcnemar(b_error: list[bool], d_error: list[bool]) -> dict:
    """Exact McNemar test for paired B/D outcomes.

    The two pipelines answer the same questions. Fisher's exact test assumes
    independent samples and is therefore not appropriate here. The exact
    binomial test on discordant pairs is McNemar's exact test.
    """
    both_error = sum(b and d for b, d in zip(b_error, d_error))
    b_only = sum(b and not d for b, d in zip(b_error, d_error))
    d_only = sum(not b and d for b, d in zip(b_error, d_error))
    both_ok = sum(not b and not d for b, d in zip(b_error, d_error))
    discordant = b_only + d_only
    p_value = (
        binomtest(min(b_only, d_only), n=discordant, p=0.5, alternative="two-sided").pvalue
        if discordant else 1.0
    )
    total = len(b_error)
    return {
        "paired_contingency_table": {
            "both_error": both_error,
            "B_only_error": b_only,
            "D_only_error": d_only,
            "both_ok": both_ok,
        },
        "rate_B": (both_error + b_only) / total if total else None,
        "rate_D": (both_error + d_only) / total if total else None,
        "n_discordant": discordant,
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
        "test": "McNemar exact (binomial)",
    }


def main():
    parser = argparse.ArgumentParser(description="Test H1 apparié (échantillon commun B et D).")
    parser.add_argument("--error_classification_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ecd = Path(args.error_classification_dir)
    b = load(ecd / "execution_based_B.json")
    d = load(ecd / "execution_based_D.json")

    attempted_b = {qid for qid, r in b.items() if r["category"] not in EXCLUDED_CATEGORIES}
    attempted_d = {qid for qid, r in d.items() if r["category"] not in EXCLUDED_CATEGORIES}
    paired = attempted_b & attempted_d  # les DEUX ont tenté

    print(f"Questions tentées par B seul  : {len(attempted_b)}")
    print(f"Questions tentées par D seul  : {len(attempted_d)}")
    print(f"Intersection (échantillon apparié) : {len(paired)}")
    print(f"  -> tentées par D mais pas B (perdues du test standard) : {len(attempted_d - attempted_b)}")
    print(f"  -> tentées par B mais pas D : {len(attempted_b - attempted_d)}")
    print()

    n = len(paired)
    match_b = sum(1 for qid in paired if b[qid]["category"] == "match")
    match_d = sum(1 for qid in paired if d[qid]["category"] == "match")

    struct_b = [b[qid]["category"] in STRUCTURAL_CATEGORIES for qid in paired]
    struct_d = [d[qid]["category"] in STRUCTURAL_CATEGORIES for qid in paired]
    sem_b = [b[qid]["category"] in SEMANTIC_CATEGORIES for qid in paired]
    sem_d = [d[qid]["category"] in SEMANTIC_CATEGORIES for qid in paired]

    result = {
        "n_paired": n,
        "n_attempted_B_only": len(attempted_b),
        "n_attempted_D_only": len(attempted_d),
        "match_B": match_b, "match_D": match_d,
        "structural_test_paired": run_mcnemar(struct_b, struct_d),
        "semantic_test_paired": run_mcnemar(sem_b, sem_d),
    }

    struct_ok = not result["structural_test_paired"]["significant_at_0.05"]
    sem_ok = (
        result["semantic_test_paired"]["significant_at_0.05"]
        and result["semantic_test_paired"]["rate_D"] < result["semantic_test_paired"]["rate_B"]
    )
    result["H1_supported_paired"] = bool(struct_ok and sem_ok)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["H1_supported_paired"]:
        print("\n>>> H1 SOUTENUE sur l'échantillon apparié (biais de sélection éliminé).")
    else:
        print("\n>>> H1 NON SOUTENUE même sur l'échantillon apparié — voir détail.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nRésultat écrit dans {out_path}")


if __name__ == "__main__":
    main()
