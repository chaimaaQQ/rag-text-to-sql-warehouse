"""Create a canonical question set with exact duplicate records removed.

The original input is never modified.  A repeated ``question_id`` is accepted
only when all experiment-relevant fields are identical; otherwise the command
stops so that an ambiguous data problem cannot silently alter the study.
"""

import argparse
import hashlib
import json
from pathlib import Path


FIELDS = ("question_id", "db_id", "question", "evidence", "sql_gold", "difficulty")


def fingerprint(item: dict) -> str:
    payload = {field: item.get(field) for field in FIELDS}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize(items: list[dict]) -> tuple[list[dict], list[dict]]:
    unique: list[dict] = []
    first_by_id: dict[str, tuple[str, int]] = {}
    duplicates: list[dict] = []

    for position, item in enumerate(items):
        question_id = item.get("question_id")
        if not question_id:
            raise ValueError(f"Entrée {position} sans question_id.")

        digest = fingerprint(item)
        previous = first_by_id.get(question_id)
        if previous is None:
            first_by_id[question_id] = (digest, position)
            unique.append(item)
            continue

        previous_digest, previous_position = previous
        if digest != previous_digest:
            raise ValueError(
                f"question_id dupliqué mais contenu différent : {question_id} "
                f"(positions {previous_position} et {position})."
            )
        duplicates.append({
            "question_id": question_id,
            "kept_source_position": previous_position,
            "removed_source_position": position,
            "reason": "exact_duplicate",
        })

    return unique, duplicates


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retire les doublons exacts des questions du protocole.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)
    if output_path.resolve() == input_path.resolve():
        parser.error("--output doit être différent de --input pour préserver la source.")

    with input_path.open(encoding="utf-8") as stream:
        items = json.load(stream)
    if not isinstance(items, list):
        parser.error("Le fichier d'entrée doit contenir une liste JSON.")

    unique, duplicates = normalize(items)
    write_json(output_path, unique)
    write_json(report_path, {
        "source_path": str(input_path),
        "source_count": len(items),
        "normalized_count": len(unique),
        "removed_exact_duplicates": len(duplicates),
        "duplicates": duplicates,
    })
    print(f"{len(items)} entrées source -> {len(unique)} questions uniques.")
    print(f"Rapport écrit dans {report_path}")


if __name__ == "__main__":
    main()
