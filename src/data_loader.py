
import argparse
import json
from pathlib import Path


def load_raw_questions(input_path: str) -> list[dict]:
    """Charge le fichier BIRD brut. Gère le cas d'une liste directe ou d'un
    dict enveloppant une clé 'data'/'questions'."""
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("data", "questions", "items"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
    raise ValueError(
        "Format inattendu : impossible de trouver une liste de questions "
        "dans le fichier fourni."
    )


def normalize_question(item: dict, index: int) -> dict:
    """Convertit une entrée brute BIRD vers le format JSON unifié du projet."""
    question_id = item.get("question_id")
    if question_id is None:
        question_id = f"bird_mini_{index:04d}"
    else:
        question_id = f"bird_mini_{question_id}"

    db_id = item.get("db_id", "")

    return {
        "question_id": question_id,
        "db_id": db_id,
        "question": item.get("question", "").strip(),
        "evidence": item.get("evidence", "").strip(),
        # BIRD utilise la clé "SQL" pour la requête de référence
        "sql_gold": item.get("SQL", item.get("sql", "")).strip(),
        "difficulty": item.get("difficulty", "unknown"),
        "schema_ref": f"schemas/{db_id}.json" if db_id else "",
        "metadata": {
            "source": "BIRD-Mini-Dev",
            "split": "dev",
        },
    }


def build_unified_dataset(input_path: str) -> list[dict]:
    raw_items = load_raw_questions(input_path)
    return [normalize_question(item, i) for i, item in enumerate(raw_items)]


def save_unified(questions: list[dict], output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"{len(questions)} questions écrites dans {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Charge BIRD Mini-Dev brut et produit le format JSON unifié."
    )
    parser.add_argument("--input", required=True, help="Chemin du fichier BIRD brut (ex: dev.json)")
    parser.add_argument("--output", required=True, help="Chemin de sortie (ex: data/processed/questions.json)")
    args = parser.parse_args()

    questions = build_unified_dataset(args.input)

    missing_db = sum(1 for q in questions if not q["db_id"])
    if missing_db:
        print(f"Attention : {missing_db} question(s) sans db_id.")

    save_unified(questions, args.output)


if __name__ == "__main__":
    main()