"""
sql_validator.py — Module 5, volet structurel (Étudiant A)

Vérifie qu'une requête SQL générée est structurellement cohérente avec le
schéma réel de la base : la requête est-elle syntaxiquement valide, les
tables référencées existent-elles, les colonnes référencées existent-elles
dans les bonnes tables ?

Ce fichier ne juge PAS si la requête répond correctement à la question
métier (ça, c'est le volet sémantique d'Étudiant B) — uniquement sa
cohérence structurelle avec le schéma.

Réutilise load_schema_lookup() de evaluator.py pour éviter de dupliquer la
logique de chargement des schémas.

Expose aussi une classe SQLValidator (interface simple .validate(sql)),
requise par llm_client.py (B) qui fait `from sql_validator import SQLValidator`
et `self.validator = SQLValidator()`. Cette classe est un wrapper au-dessus
de validate_query() ci-dessous — même logique, interface différente pour
s'intégrer au pipeline LLMClient → SQLValidator de B.

Dépendance :
    pip install sqlglot

Usage (une seule requête, test rapide) :
    python sql_validator.py check --sql "SELECT District FROM schools" --db_id california_schools --schemas_dir data\\schemas

Usage (lot de requêtes générées par le LLM, format [{question_id, db_id, sql_generated}, ...]) :
    python sql_validator.py validate-batch --generated data\\results\\generated_sql.json \
        --schemas_dir data\\schemas --output data\\evaluation\\structural_validation.json

Usage (auto-test sur sql_gold — doit donner ~100% de valid, sinon le validateur a un bug) :
    python sql_validator.py validate-batch --generated data\\processed\\questions.json \
        --schemas_dir data\\schemas --output data\\evaluation\\structural_validation_gold.json --sql_field sql_gold
"""

import argparse
import json
import re
from pathlib import Path

import sqlglot
from sqlglot import exp

from evaluator import load_schema_lookup


FORBIDDEN_KEYWORDS = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"]


# ============================================================================
# 1. Validation structurelle d'une requête
# ============================================================================

def validate_query(sql: str, db_id: str, schema_lookup: dict, dialect: str = "sqlite") -> dict:
    """Retourne un rapport structurel complet pour une requête SQL donnée.

    error_type possibles (par ordre de priorité) :
        "syntax_error"      — sqlglot n'a pas pu parser la requête
        "db_unknown"        — db_id absent du schema_lookup
        "table_not_found"   — une table référencée n'existe pas dans le schéma
        "column_not_found"  — une colonne référencée n'existe pas dans sa table
        "column_unresolved" — colonne ambiguë, impossible à rattacher à une table
        None                — requête structurellement valide
    """
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

    # --- 0. SQL vide : ce n'est PAS une erreur de syntaxe, c'est l'absence
    # de toute tentative (typiquement un refus volontaire du LLM du type
    # ADDITIONAL_INFORMATION_REQUIRED, cf. sql_generator.py). À ne jamais
    # confondre avec un vrai SQL syntaxiquement invalide dans les métriques.
    if not sql or not sql.strip():
        report["error_type"] = "empty_sql"
        report["error_detail"] = "Aucune requête SQL fournie (sql_generated vide)."
        return report

    # --- 1. Parsing syntaxique ---
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
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


# ============================================================================
# 1bis. Classe SQLValidator — interface attendue par llm_client.py (B)
# ============================================================================

DEFAULT_SCHEMAS_DIR = "data/schemas"  # relatif à l'endroit d'exécution du script appelant


class SQLValidator:
    """Validateur SQL unifié — fusionne la version structurelle multi-bases
    (Étudiant A, sur les schémas .sqlite de BIRD) et les vérifications de
    sécurité de la version de B (SELECT uniquement, commandes interdites),
    en gardant la résolution d'alias (absente de la version de B, qui
    rejetait à tort tout SQL utilisant des alias de table).

    Deux sources de schéma possibles, au choix :
      - schemas_dir=...   : schémas extraits des .sqlite (BIRD, multi-bases) — track A
      - glossary_path=...  : glossaire JSON à plat (une seule base, ex TPC-DS) — track B,
                              même format que data/business_docs/glossary.json

    Si aucune des deux n'est fournie ou trouvée, la validation se limite à
    SELECT-only + syntaxe (ne bloque jamais faute de schéma disponible).
    """

    def __init__(self, schemas_dir: str | None = None, glossary_path: str | None = "data/business_docs/glossary.json", dialect: str = "postgres"):
        self.dialect = dialect
        self.schema_lookup = {}

        if schemas_dir:
            # Track A explicite : priorité au schéma BIRD si fourni
            try:
                self.schema_lookup = load_schema_lookup(schemas_dir)
            except Exception:
                self.schema_lookup = {}
        elif glossary_path and Path(glossary_path).exists():
            # Comportement par défaut de B : glossaire TPC-DS, chemin par défaut inchangé
            self.schema_lookup = self._load_glossary(glossary_path)
        # sinon : aucun schéma disponible -> validation SELECT-only + syntaxe uniquement

    @staticmethod
    def _load_glossary(glossary_path: str) -> dict:
        """Charge un glossaire à plat (format TPC-DS de B) et le range sous
        une clé unique '__default__', pour rester compatible avec le format
        {db_id: {table: [colonnes]}} utilisé par validate_query()."""
        with open(glossary_path, "r", encoding="utf-8") as f:
            glossary = json.load(f)

        tables = {}
        for doc in glossary:
            table_name = doc.get("table")
            if not table_name:
                continue
            tables[table_name.lower()] = [c.lower() for c in doc.get("columns", [])]

        return {"__default__": tables}

    def validate(self, sql: str, db_id: str | None = None) -> dict:
        if not sql or not sql.strip():
            return {"valid": False, "reason": "Requête SQL vide.", "tables": [], "columns": []}

        cleaned_sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE).strip()

        if not re.match(r"^SELECT\b", cleaned_sql, re.IGNORECASE):
            return {"valid": False, "reason": "Seules les requêtes SELECT sont autorisées.", "tables": [], "columns": []}

        for keyword in FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", cleaned_sql, re.IGNORECASE):
                return {"valid": False, "reason": f"Commande interdite détectée : {keyword}", "tables": [], "columns": []}

        # Aucun schéma chargé : on ne bloque pas, on vérifie juste la syntaxe
        if not self.schema_lookup:
            try:
                sqlglot.parse_one(cleaned_sql, dialect=self.dialect)
                return {
                    "valid": True,
                    "reason": "Syntaxe valide (aucun schéma chargé : vérification structurelle ignorée).",
                    "tables": [],
                    "columns": [],
                }
            except Exception as e:
                return {"valid": False, "reason": f"Erreur de syntaxe : {e}", "tables": [], "columns": []}

        if db_id is None:
            if "__default__" in self.schema_lookup:
                db_id = "__default__"  # glossaire : une seule base implicite
            elif len(self.schema_lookup) == 1:
                db_id = next(iter(self.schema_lookup))
            else:
                return {
                    "valid": False,
                    "reason": "db_id requis : plusieurs schémas chargés, impossible de le déduire automatiquement.",
                    "tables": [],
                    "columns": [],
                }

        report = validate_query(cleaned_sql, db_id, self.schema_lookup, dialect=self.dialect)
        return {
            "valid": report["is_valid"],
            "reason": report["error_detail"] or "Structurellement valide.",
            "tables": report["tables_used"],
            "columns": report["columns_used"],
        }


# ============================================================================
# 2. Validation en lot
# ============================================================================

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


# ============================================================================
# 3. CLI
# ============================================================================

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