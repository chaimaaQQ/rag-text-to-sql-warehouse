"""
hypothesis_h1_test_v2.py — Test de l'hypothèse H1, VERSION RÉVISÉE

Utilise la classification erreurs structurelles/sémantiques basée sur
l'EXÉCUTION SQLITE RÉELLE (classify_pipeline_errors.py), au lieu de :
  - structural_X.json (validation statique sqlglot, ne détecte que le
    parsing/existence table-colonne, pas les erreurs d'exécution comme
    "no such function", "misuse of aggregate", etc.)
  - semantic_X.json (heuristique lexicale approximative, jamais validée
    manuellement)

Définition (cohérente avec la section 2.2 du cahier des charges) :
  - Erreur STRUCTURELLE : le SQL généré ne s'exécute pas du tout
    correctement (colonne/table inconnue, ambiguïté, syntaxe, fonction
    non supportée, mauvais usage d'agrégat...)
  - Erreur SÉMANTIQUE : le SQL s'exécute sans erreur mais renvoie un
    résultat différent de sql_gold (mauvaise règle métier / mauvais
    filtre / mauvaise formule de KPI)
  - Exclus des deux : refus (insufficient_context) et SQL vide (empty_sql)

Prérequis : avoir lancé classify_pipeline_errors.py --all au préalable
(génère results/error_classification/execution_based_{A,B,C,D}.json).

Usage:
    python hypothesis_h1_test_v2.py \
        --error_classification_dir ..\\results\\error_classification \
        --output ..\\results\\hypothesis_H1\\h1_test_result_v2.json
"""

import argparse
import json
from pathlib import Path

from scipy.stats import fisher_exact

from classify_pipeline_errors import STRUCTURAL_CATEGORIES, SEMANTIC_CATEGORIES, EXCLUDED_CATEGORIES


def load(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rates(per_question: list[dict]) -> dict:
    attempted = [r for r in per_question if r["category"] not in EXCLUDED_CATEGORIES]
    n_attempted = len(attempted)
    n_struct = sum(1 for r in attempted if r["category"] in STRUCTURAL_CATEGORIES)
    n_sem = sum(1 for r in attempted if r["category"] in SEMANTIC_CATEGORIES)
    n_match = sum(1 for r in attempted if r["category"] == "match")
    n_refusal = len(per_question) - n_attempted
    return {
        "n_total": len(per_question),
        "n_refusal_or_empty": n_refusal,
        "n_attempted": n_attempted,
        "n_match": n_match,
        "n_structural_errors": n_struct,
        "n_semantic_errors": n_sem,
    }


def run_fisher(errors_b: int, total_b: int, errors_d: int, total_d: int) -> dict:
    table = [[errors_b, total_b - errors_b], [errors_d, total_d - errors_d]]
    odds_ratio, p_value = fisher_exact(table)
    return {
        "contingency_table": {"B": {"errors": errors_b, "ok": total_b - errors_b},
                               "D": {"errors": errors_d, "ok": total_d - errors_d}},
        "rate_B": (errors_b / total_b) if total_b else None,
        "rate_D": (errors_d / total_d) if total_d else None,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def main():
    parser = argparse.ArgumentParser(description="Test H1 (v2, basé sur l'exécution SQLite réelle).")
    parser.add_argument("--error_classification_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ecd = Path(args.error_classification_dir)
    b_data = load(ecd / "execution_based_B.json")
    d_data = load(ecd / "execution_based_D.json")

    rates_b = rates(b_data)
    rates_d = rates(d_data)

    print("B :", json.dumps(rates_b, ensure_ascii=False, indent=2))
    print("D :", json.dumps(rates_d, ensure_ascii=False, indent=2))

    result = {
        "rates_B": rates_b,
        "rates_D": rates_d,
        "structural_test": run_fisher(
            rates_b["n_structural_errors"], rates_b["n_attempted"],
            rates_d["n_structural_errors"], rates_d["n_attempted"],
        ),
        "semantic_test": run_fisher(
            rates_b["n_semantic_errors"], rates_b["n_attempted"],
            rates_d["n_semantic_errors"], rates_d["n_attempted"],
        ),
        "refusal_test_informatif": run_fisher(
            rates_b["n_refusal_or_empty"], rates_b["n_total"],
            rates_d["n_refusal_or_empty"], rates_d["n_total"],
        ),
    }

    struct_ok = not result["structural_test"]["significant_at_0.05"]
    sem_ok = (
        result["semantic_test"]["significant_at_0.05"]
        and result["semantic_test"]["rate_D"] < result["semantic_test"]["rate_B"]
    )
    result["H1_supported"] = bool(struct_ok and sem_ok)

    print()
    print(json.dumps({
        "structural_test": result["structural_test"],
        "semantic_test": result["semantic_test"],
        "refusal_test_informatif": result["refusal_test_informatif"],
        "H1_supported": result["H1_supported"],
    }, ensure_ascii=False, indent=2))

    if result["H1_supported"]:
        print("\n>>> H1 SOUTENUE (version exécution réelle).")
    else:
        print("\n>>> H1 NON SOUTENUE en l'état (version exécution réelle) — voir détail ci-dessus.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nRésultat écrit dans {out_path}")


if __name__ == "__main__":
    main()