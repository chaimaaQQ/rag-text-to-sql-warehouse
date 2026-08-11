"""
sql_validator.py — Module 5, volet structurel


"""

import argparse
import json
from pathlib import Path

import sqlglot
from sqlglot import exp

from evaluator import load_schema_lookup


# 1. Validation structurelle d'une requête


def validate_query(sql: str, db_id: str, schema_lookup: dict) -> dict:
    
    report = {
        "is_valid": False,
        "error_type": None,
        "error_detail": None,
        "tables_used": [],
        "tables_missing": [],
        "columns_used": [],
        "columns_missing": [],
    }

    if db_id not in schema_lookup:
        report["error_type"] = "db_unknown"
        report["error_detail"] = f"db_id '{db_id}' absent des schémas chargés"
        return report

    db_tables = schema_lookup[db_id]  # {table_name_lower: [colonnes]}

    # --- 1. Parsing syntaxique ---
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception as e:
        report["error_type"] = "syntax_error"
        report["error_detail"] = str(e)
        return report

    # --- 2. Vérification des tables ---
    tables_used, alias_to_table = set(), {}
    for table_node in parsed.find_all(exp.Table):
        table_name = table_node.name.lower()
        tables_used.add(table_name)
        if table_node.alias:
            alias_to_table[table_node.alias.lower()] = table_name

    tables_missing = sorted(t for t in tables_used if t not in db_tables)
    report["tables_used"] = sorted(tables_used)
    report["tables_missing"] = tables_missing

    if tables_missing:
        report["error_type"] = "table_not_found"
        report["error_detail"] = f"Table(s) inexistante(s) dans le schéma : {', '.join(tables_missing)}"
        return report

    # --- 3. Vérification des colonnes ---
    columns_used, columns_missing, unresolved = set(), set(), []

    for col_node in parsed.find_all(exp.Column):
        col_name = col_node.name
        table_ref = col_node.table.lower() if col_node.table else None
        resolved_table = alias_to_table.get(table_ref, table_ref)

        if resolved_table is None and len(tables_used) == 1:
            resolved_table = next(iter(tables_used))

        if resolved_table is None:
            candidates = [t for t in tables_used if col_name in db_tables.get(t, [])]
            if len(candidates) == 1:
                resolved_table = candidates[0]
            else:
                unresolved.append(col_name)
                continue

        columns_used.add(f"{resolved_table}.{col_name}")
        if col_name not in db_tables.get(resolved_table, []):
            columns_missing.add(f"{resolved_table}.{col_name}")

    report["columns_used"] = sorted(columns_used)
    report["columns_missing"] = sorted(columns_missing)

    if columns_missing:
        report["error_type"] = "column_not_found"
        report["error_detail"] = f"Colonne(s) inexistante(s) : {', '.join(sorted(columns_missing))}"
        return report

    if unresolved:
        report["error_type"] = "column_unresolved"
        report["error_detail"] = f"Colonne(s) ambiguë(s), non rattachée(s) à une table : {', '.join(unresolved)}"
        return report

    report["is_valid"] = True
    return report


# 2. Validation en lot

def validate_batch(generated_path: str, schemas_dir: str, sql_field: str = "sql_generated") -> list[dict]:
    with open(generated_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    schema_lookup = load_schema_lookup(schemas_dir)

    results = []
    for item in items:
        report = validate_query(item[sql_field], item["db_id"], schema_lookup)
        results.append({
            "question_id": item["question_id"],
            "db_id": item["db_id"],
            **report,
        })

    n_valid = sum(r["is_valid"] for r in results)
    print(f"{n_valid}/{len(results)} requêtes structurellement valides ({100 * n_valid / len(results):.1f}%)")

    # Répartition des erreurs, utile pour l'analyse d'erreurs structurelles
    error_counts = {}
    for r in results:
        if r["error_type"]:
            error_counts[r["error_type"]] = error_counts.get(r["error_type"], 0) + 1
    if error_counts:
        print("Répartition des erreurs :", json.dumps(error_counts, ensure_ascii=False))

    return results


def save_results(results: list[dict], output_path: str):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Rapport écrit dans {out_path}")


# 3. CLI


def main():
    parser = argparse.ArgumentParser(description="Validation structurelle du SQL (tables/colonnes vs schéma).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_p = subparsers.add_parser("check", help="Valide une seule requête (test rapide)")
    check_p.add_argument("--sql", required=True)
    check_p.add_argument("--db_id", required=True)
    check_p.add_argument("--schemas_dir", required=True)

    batch_p = subparsers.add_parser("validate-batch", help="Valide un lot de requêtes")
    batch_p.add_argument("--generated", required=True, help="JSON avec [{question_id, db_id, <sql_field>}, ...]")
    batch_p.add_argument("--schemas_dir", required=True)
    batch_p.add_argument("--output", required=True)
    batch_p.add_argument("--sql_field", default="sql_generated", help="Nom du champ SQL à valider (ex: sql_gold pour un auto-test)")

    args = parser.parse_args()

    if args.command == "check":
        schema_lookup = load_schema_lookup(args.schemas_dir)
        report = validate_query(args.sql, args.db_id, schema_lookup)
        print(json.dumps(report, ensure_ascii=False, indent=2))

    elif args.command == "validate-batch":
        results = validate_batch(args.generated, args.schemas_dir, args.sql_field)
        save_results(results, args.output)


if __name__ == "__main__":
    main()