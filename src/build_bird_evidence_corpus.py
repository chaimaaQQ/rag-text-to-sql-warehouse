import json
from pathlib import Path

INPUT_FILE = Path("data/raw/bird/mini_dev_sqlite.json")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    bird_data = json.load(f)

corpus = []
seen = set()
for sample in bird_data:

    question_id = sample["question_id"]
    db_id = sample["db_id"]
    evidence = sample["evidence"]

    document = {
        "id": question_id,
        "db_id": db_id,
        "text": evidence
    }

    corpus.append(document)

print(corpus[0])
print(len(corpus))


OUTPUT_DIR = Path("data/bird_evidence_corpus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "bird_evidence_corpus.json"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(corpus, f, indent=4, ensure_ascii=False)

print(f"Corpus sauvegardé dans : {OUTPUT_FILE}")
print(f"Nombre de documents : {len(corpus)}")