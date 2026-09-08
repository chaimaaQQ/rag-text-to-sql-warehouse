"""Create and score double-annotation sheets for SQL errors.

Automatic execution outcomes are useful, but they do not by themselves tell
whether a wrong result is caused by a business-rule error or an executable
but invalid join. This utility supplies the manual A/B validation required by
the PFA brief and reports Cohen's kappa in addition to raw agreement.
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


LABELS = ("structural", "semantic", "ambiguous", "correct", "other")


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def write_sheet(rows: list[dict], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "item_id", "question_id", "pipeline", "question", "evidence", "sql_gold",
        "sql_generated", "execution_status", "execution_error", "annotator_a", "annotator_b", "comment",
    )
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample(args) -> None:
    questions = {row["question_id"]: row for row in load_json(args.questions)}
    candidates = []
    for pipeline, generated_path, execution_path in (
        ("B", args.generated_b, args.execution_b),
        ("D", args.generated_d, args.execution_d),
    ):
        generated = {row["question_id"]: row for row in load_json(generated_path)}
        execution = load_json(execution_path)["details"]
        for result in execution:
            if result.get("execution_match") is None:
                continue
            generated_row = generated.get(result["question_id"], {})
            if generated_row.get("insufficient_context") or not generated_row.get("sql_generated"):
                continue
            question = questions.get(result["question_id"], {})
            candidates.append({
                "item_id": f"{pipeline}_{result['question_id']}",
                "question_id": result["question_id"],
                "pipeline": pipeline,
                "question": question.get("question", ""),
                "evidence": question.get("evidence", ""),
                "sql_gold": question.get("sql_gold", ""),
                "sql_generated": generated_row.get("sql_generated", ""),
                "execution_status": "correct" if result.get("execution_match") else "incorrect",
                "execution_error": result.get("error") or "",
                "annotator_a": "",
                "annotator_b": "",
                "comment": "",
            })

    if len(candidates) < args.n:
        raise ValueError(f"Seulement {len(candidates)} requetes annotables, moins que --n={args.n}.")
    random.Random(args.seed).shuffle(candidates)
    write_sheet(candidates[:args.n], args.output)
    print(f"Feuille de {args.n} requetes ecrite dans {args.output}.")
    print("Chaque annotateur doit utiliser uniquement : " + ", ".join(LABELS))


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    n = len(labels_a)
    if not n:
        return None
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    counts_a, counts_b = Counter(labels_a), Counter(labels_b)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in set(labels_a) | set(labels_b))
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)


def score(args) -> None:
    with open(args.sheet, "r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    completed = [
        row for row in rows
        if row["annotator_a"].strip().lower() in LABELS and row["annotator_b"].strip().lower() in LABELS
    ]
    if len(completed) < args.min_completed:
        raise ValueError(
            f"{len(completed)} annotations completes ; au moins {args.min_completed} sont requises."
        )
    labels_a = [row["annotator_a"].strip().lower() for row in completed]
    labels_b = [row["annotator_b"].strip().lower() for row in completed]
    disagreements = [
        {"item_id": row["item_id"], "a": row["annotator_a"], "b": row["annotator_b"], "comment": row["comment"]}
        for row in completed if row["annotator_a"].strip().lower() != row["annotator_b"].strip().lower()
    ]
    report = {
        "n_completed": len(completed),
        "raw_agreement": sum(a == b for a, b in zip(labels_a, labels_b)) / len(completed),
        "cohens_kappa": cohens_kappa(labels_a, labels_b),
        "labels_annotator_a": dict(Counter(labels_a)),
        "labels_annotator_b": dict(Counter(labels_b)),
        "disagreements": disagreements,
        "label_set": list(LABELS),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation inter-annotateur des erreurs SQL.")
    sub = parser.add_subparsers(dest="command", required=True)
    sample_parser = sub.add_parser("sample", help="Cree une feuille CSV a annoter par A et B.")
    sample_parser.add_argument("--questions", required=True)
    sample_parser.add_argument("--generated_b", required=True)
    sample_parser.add_argument("--execution_b", required=True)
    sample_parser.add_argument("--generated_d", required=True)
    sample_parser.add_argument("--execution_d", required=True)
    sample_parser.add_argument("--output", required=True)
    sample_parser.add_argument("--n", type=int, default=60)
    sample_parser.add_argument("--seed", type=int, default=42)

    score_parser = sub.add_parser("score", help="Calcule accord brut et kappa apres annotation.")
    score_parser.add_argument("--sheet", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--min_completed", type=int, default=30)
    args = parser.parse_args()
    sample(args) if args.command == "sample" else score(args)


if __name__ == "__main__":
    main()
