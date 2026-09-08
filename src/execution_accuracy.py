"""
execution_accuracy.py — Module 6 (partie manquante) : Execution Accuracy

Métrique PRIMAIRE du projet (section 11 du cahier des charges) : exécute
le SQL généré ET le sql_gold sur la vraie base SQLite, compare les
résultats retournés (pas juste "le SQL est-il bien formé").

Prérequis : dev_databases/<db_id>/<db_id>.sqlite doit exister pour
chaque db_id (téléchargement officiel BIRD Mini-Dev, voir minidev.zip :
https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip).

Règles de comparaison (standard BIRD / littérature Text-to-SQL) :
  - Comparaison en tant qu'ENSEMBLE de lignes (set), pas de liste ordonnée,
    SAUF si sql_gold contient un ORDER BY explicite (auquel cas l'ordre
    compte).
  - Les valeurs numériques sont arrondies à `--round_decimals` décimales
    avant comparaison, pour éviter les faux négatifs dus à des différences
    d'arrondi flottant (ex: 33.333333 vs 33.33333333).
  - Un timeout (`--timeout`) protège contre les requêtes générées qui
    boucleraient indéfiniment (ex: produit cartésien géant).

Usage:
    python execution_accuracy.py --generated ..\\results\\pipelines_A_B_C_D\\pipeline_D\\generated_sql.json \
        --questions ..\\data\\processed\\questions.json \
        --databases_dir ..\\data\\raw\\dev_databases \
        --output ..\\results\\pipelines_A_B_C_D\\pipeline_D\\execution_accuracy.json
"""

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path


def has_order_by(sql: str) -> bool:
    """Détecte un ORDER BY au niveau du SELECT principal (approximation
    simple mais suffisante ici : suffisant pour distinguer les questions
    'classement/tri' des agrégations classiques)."""
    return bool(re.search(r"\border\s+by\b", sql or "", re.IGNORECASE))


def normalize_value(v, round_decimals: int):
    if isinstance(v, float):
        return round(v, round_decimals)
    if isinstance(v, str):
        return v.strip()
    return v


def normalize_rows(rows: list[tuple], round_decimals: int) -> list[tuple]:
    return [tuple(normalize_value(v, round_decimals) for v in row) for row in rows]


def execute_sql(db_path: str, sql: str, timeout: float) -> tuple[list[tuple] | None, str | None]:
    """Exécute sql sur db_path. Retourne (rows, error) — l'un des deux est None."""
    if not sql or not sql.strip():
        return None, "SQL vide (aucune tentative)."
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
        deadline = time.monotonic() + timeout
        # Le paramètre ``timeout`` de sqlite3 ne couvre que les verrous. Le
        # progress handler applique aussi cette même limite aux requêtes
        # coûteuses (p. ex. des jointures cartésiennes générées par le LLM).
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 10_000)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return rows, None
    except Exception as e:
        return None, str(e)
    finally:
        if conn is not None:
            conn.close()


def compare_results(gold_rows: list[tuple], gen_rows: list[tuple], ordered: bool,
                     round_decimals: int) -> bool:
    gold_norm = normalize_rows(gold_rows, round_decimals)
    gen_norm = normalize_rows(gen_rows, round_decimals)
    if ordered:
        return gold_norm == gen_norm
    # Comparaison en multiset (pas juste set, pour ne pas ignorer les doublons)
    from collections import Counter
    return Counter(gold_norm) == Counter(gen_norm)


def evaluate_pipeline(generated_path: str, questions_path: str, databases_dir: str,
                       round_decimals: int, timeout: float) -> dict:
    with open(generated_path, "r", encoding="utf-8") as f:
        generated = {item["question_id"]: item for item in json.load(f)}
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = {q["question_id"]: q for q in json.load(f)}

    databases_dir = Path(databases_dir)
    results = []

    for qid, gen in generated.items():
        q = questions.get(qid)
        if q is None:
            continue

        db_id = gen["db_id"]
        db_path = databases_dir / db_id / f"{db_id}.sqlite"

        entry = {"question_id": qid, "db_id": db_id}

        if not db_path.exists():
            entry["execution_match"] = None
            entry["error"] = f"Base introuvable : {db_path}"
            results.append(entry)
            continue

        gold_sql = q.get("sql_gold", "")
        gen_sql = gen.get("sql_generated", "")

        gold_rows, gold_err = execute_sql(str(db_path), gold_sql, timeout)
        gen_rows, gen_err = execute_sql(str(db_path), gen_sql, timeout)

        if gold_err:
            # sql_gold qui ne s'exécute pas = problème de données de référence,
            # à signaler séparément (ne doit normalement jamais arriver).
            entry["execution_match"] = None
            entry["error"] = f"GOLD SQL EN ECHEC (à investiguer) : {gold_err}"
        elif gen_err:
            entry["execution_match"] = False
            entry["error"] = f"SQL généré en échec à l'exécution : {gen_err}"
        else:
            ordered = has_order_by(gold_sql)
            match = compare_results(gold_rows, gen_rows, ordered, round_decimals)
            entry["execution_match"] = match
            entry["error"] = None

        results.append(entry)

    valid_comparisons = [r for r in results if r["execution_match"] is not None]
    n_match = sum(1 for r in valid_comparisons if r["execution_match"])
    n_total_comparable = len(valid_comparisons)
    n_total = len(results)

    summary = {
        "execution_accuracy": (n_match / n_total_comparable) if n_total_comparable else None,
        "n_match": n_match,
        "n_comparable": n_total_comparable,
        "n_total": n_total,
        "n_db_missing_or_gold_failed": n_total - n_total_comparable,
    }

    return {"summary": summary, "details": results}


def main():
    parser = argparse.ArgumentParser(description="Calcule l'Execution Accuracy (métrique primaire, section 11).")
    parser.add_argument("--generated", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--databases_dir", required=True,
                         help="Dossier contenant dev_databases/<db_id>/<db_id>.sqlite")
    parser.add_argument("--output", required=True)
    parser.add_argument("--round_decimals", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=15.0, help="Timeout par requête (secondes)")
    args = parser.parse_args()

    result = evaluate_pipeline(args.generated, args.questions, args.databases_dir,
                                args.round_decimals, args.timeout)

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Rapport détaillé écrit dans {out_path}")


if __name__ == "__main__":
    main()
