"""
schema_parser.py — Module 2 (base commune, exploitée par Étudiant A pour le
retrieval de schéma)

Extrait, pour chaque base .sqlite de BIRD Mini-Dev, la liste des tables,
colonnes et clés étrangères directement via PRAGMA (plutôt que retapées à la
main), et produit un fichier JSON de schéma indexable par db_id.

Structure attendue des données brutes BIRD Mini-Dev :
    dev_databases/
        california_schools/
            california_schools.sqlite
        card_games/
            card_games.sqlite
        ...

Usage:
    python schema_parser.py --databases_dir dev_databases --output_dir data/schemas
"""

import argparse
import json
import sqlite3
from pathlib import Path


def get_tables(cursor: sqlite3.Cursor) -> list[str]:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    return [row[0] for row in cursor.fetchall()]


def get_columns(cursor: sqlite3.Cursor, table_name: str) -> list[dict]:
    cursor.execute(f'PRAGMA table_info("{table_name}");')
    # colonnes retournées : (cid, name, type, notnull, dflt_value, pk)
    return [
        {"name": row[1], "type": row[2] or "", "description": ""}
        for row in cursor.fetchall()
    ]


def get_foreign_keys(cursor: sqlite3.Cursor, table_name: str) -> list[dict]:
    cursor.execute(f'PRAGMA foreign_key_list("{table_name}");')
    # colonnes retournées : (id, seq, table, from, to, on_update, on_delete, match)
    return [
        {"from": f"{table_name}.{row[3]}", "to": f"{row[2]}.{row[4]}"}
        for row in cursor.fetchall()
    ]


def extract_schema_from_sqlite(db_path: Path, db_id: str) -> dict:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    tables = []
    foreign_keys = []
    for table_name in get_tables(cursor):
        tables.append(
            {
                "name": table_name,
                "description": "",
                "columns": get_columns(cursor, table_name),
            }
        )
        foreign_keys.extend(get_foreign_keys(cursor, table_name))

    conn.close()

    return {
        "db_id": db_id,
        "tables": tables,
        "foreign_keys": foreign_keys,
    }


def process_all_databases(databases_dir: str, output_dir: str) -> None:
    databases_dir = Path(databases_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_folders = sorted(p for p in databases_dir.iterdir() if p.is_dir())
    if not db_folders:
        print(f"Aucun dossier de base trouvé dans {databases_dir}")
        return

    for folder in db_folders:
        db_id = folder.name
        sqlite_path = folder / f"{db_id}.sqlite"
        if not sqlite_path.exists():
            sqlite_files = list(folder.glob("*.sqlite"))
            if not sqlite_files:
                print(f"Attention : aucun .sqlite trouvé pour {db_id}, ignoré.")
                continue
            sqlite_path = sqlite_files[0]

        schema = extract_schema_from_sqlite(sqlite_path, db_id)
        out_path = output_dir / f"{db_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)

        print(f"{db_id} : {len(schema['tables'])} tables, {len(schema['foreign_keys'])} clés étrangères -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extrait le schéma indexable des bases .sqlite de BIRD Mini-Dev."
    )
    parser.add_argument("--databases_dir", required=True, help="Dossier contenant un sous-dossier par base (ex: dev_databases)")
    parser.add_argument("--output_dir", required=True, help="Dossier de sortie pour les schémas JSON (ex: data/schemas)")
    args = parser.parse_args()

    process_all_databases(args.databases_dir, args.output_dir)


if __name__ == "__main__":
    main()