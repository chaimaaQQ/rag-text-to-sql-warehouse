"""
hypothesis_h1_test.py — Test de l'hypothèse H1 (section 2.2 et 7.5)

H1 : le RAG hybride (D) réduit significativement le taux d'erreurs
SEMANTIQUES par rapport au RAG schéma seul (B), SANS réduire
significativement le taux d'erreurs STRUCTURELLES (déjà corrigées par le
retrieval de schéma seul).

Prédiction testée :
  - test A : proportion d'erreurs sémantiques, B vs D -> doit être
    significativement différente (plus basse pour D)
  - test B : proportion d'erreurs structurelles, B vs D -> ne doit PAS
    différer significativement

Utilise un test exact de Fisher (adapté aux petits effectifs / cellules
proches de 0, plus prudent qu'un chi² sur ~quelques centaines de questions).

Usage:
    python hypothesis_h1_test.py \
        --structural_b ..\\data\\evaluation\\structural_B.json \
        --structural_d ..\\data\\evaluation\\structural_D.json \
        --semantic_b ..\\results\\error_classification\\semantic_B.json \
        --semantic_d ..\\results\\error_classification\\semantic_D.json \
        --output ..\\results\\hypothesis_H1\\h1_test_result.json
"""

import argparse
import json
from pathlib import Path

from scipy.stats import fisher_exact


def load(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def structural_error_rate(structural: list[dict]) -> tuple[int, int]:
    """Retourne (n_erreurs_structurelles, n_tenté).

    IMPORTANT : exclut les refus volontaires (error_type == 'empty_sql',
    correspondant à ADDITIONAL_INFORMATION_REQUIRED — cf. sql_generator.py
    et sql_validator.py) du calcul. Un refus n'est PAS une erreur
    structurelle (mauvaise table/colonne/jointure) : c'est l'absence de
    toute tentative, généralement causée par une ambiguïté métier non
    résolue (surtout marqué chez B, qui n'a pas accès à `evidence`).
    Les inclure gonflerait artificiellement l'écart B vs D sans rapport
    avec la qualité structurelle du SQL réellement généré — biaisant
    directement le test de H1 (section 2.2)."""
    attempted = [r for r in structural if r.get("error_type") != "empty_sql"]
    n_total = len(attempted)
    n_errors = sum(1 for r in attempted if r.get("is_valid") is False)
    return n_errors, n_total


def refusal_rate(structural: list[dict]) -> tuple[int, int]:
    """Retourne (n_refus, n_total) — proportion de ADDITIONAL_INFORMATION_REQUIRED
    (error_type == 'empty_sql'). Hors périmètre direct de H1, mais utile pour
    documenter pourquoi B et D diffèrent tant en Execution Accuracy globale :
    B n'a pas accès à `evidence`, donc refuse beaucoup plus souvent."""
    n_total = len(structural)
    n_refusals = sum(1 for r in structural if r.get("error_type") == "empty_sql")
    return n_refusals, n_total


def semantic_error_rate(semantic: list[dict]) -> tuple[int, int]:
    """Retourne (n_erreurs_sémantiques, n_évalué) — ne compte que les
    requêtes structurellement valides (cf. semantic_validator.py, qui
    marque is_semantic_error=None pour les autres, évitant le double
    comptage avec les erreurs structurelles)."""
    evaluated = [r for r in semantic if r.get("is_semantic_error") is not None]
    n_total = len(evaluated)
    n_errors = sum(1 for r in evaluated if r["is_semantic_error"])
    return n_errors, n_total


def run_fisher(errors_b: int, total_b: int, errors_d: int, total_d: int) -> dict:
    table = [[errors_b, total_b - errors_b], [errors_d, total_d - errors_d]]
    odds_ratio, p_value = fisher_exact(table)
    odds_ratio = float(odds_ratio)
    p_value = float(p_value)
    return {
        "contingency_table": {"B": {"errors": errors_b, "ok": total_b - errors_b},
                               "D": {"errors": errors_d, "ok": total_d - errors_d}},
        "rate_B": (errors_b / total_b) if total_b else None,
        "rate_D": (errors_d / total_d) if total_d else None,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
    }


def main():
    parser = argparse.ArgumentParser(description="Test statistique de l'hypothèse H1 (B vs D).")
    parser.add_argument("--structural_b", required=True)
    parser.add_argument("--structural_d", required=True)
    parser.add_argument("--semantic_b", required=True)
    parser.add_argument("--semantic_d", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    struct_b = load(args.structural_b)
    struct_d = load(args.structural_d)
    sem_b = load(args.semantic_b)
    sem_d = load(args.semantic_d)

    se_b, st_b = structural_error_rate(struct_b)
    se_d, st_d = structural_error_rate(struct_d)
    ref_b, reft_b = refusal_rate(struct_b)
    ref_d, reft_d = refusal_rate(struct_d)
    sem_errors_b, sem_total_b = semantic_error_rate(sem_b)
    sem_errors_d, sem_total_d = semantic_error_rate(sem_d)

    result = {
        "structural_test": run_fisher(se_b, st_b, se_d, st_d),
        "semantic_test": run_fisher(sem_errors_b, sem_total_b, sem_errors_d, sem_total_d),
        "refusal_test_informatif": run_fisher(ref_b, reft_b, ref_d, reft_d),
    }

    struct_ok = not result["structural_test"]["significant_at_0.05"]
    sem_ok = (
        result["semantic_test"]["significant_at_0.05"]
        and result["semantic_test"]["rate_D"] < result["semantic_test"]["rate_B"]
    )
    result["H1_supported"] = bool(struct_ok and sem_ok)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["H1_supported"]:
        print("\n>>> H1 SOUTENUE : baisse significative des erreurs sémantiques B->D, "
              "pas de différence significative sur les erreurs structurelles.")
    else:
        print("\n>>> H1 NON SOUTENUE en l'état — voir le détail des deux tests ci-dessus.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nRésultat écrit dans {out_path}")


if __name__ == "__main__":
    main()