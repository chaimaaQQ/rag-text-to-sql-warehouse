import json
from pathlib import Path


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

GLOSSARY_PATH = BASE_DIR / "data" / "business_docs" / "glossary.json"
KPIS_PATH = BASE_DIR / "data" / "business_docs" / "kpis.json"
RULES_PATH = BASE_DIR / "data" / "business_docs" / "business_rules.json"
BIRD_PATH = BASE_DIR / "data" / "bird_evidence_corpus" / "bird_evidence_corpus.json"

OUTPUT_DIR = BASE_DIR / "data" / "knowledge_base"
OUTPUT_PATH = OUTPUT_DIR / "knowledge_base.json"


# ==========================================================
# Utilities
# ==========================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def create_document(doc_id, doc_type, title, content, metadata=None):
    return {
        "id": f"KB{doc_id:04d}",
        "type": doc_type,
        "title": title,
        "content": content,
        "metadata": metadata or {}
    }


# ==========================================================
# Main
# ==========================================================

def main():

    glossary = load_json(GLOSSARY_PATH)
    kpis = load_json(KPIS_PATH)
    rules = load_json(RULES_PATH)
    bird = load_json(BIRD_PATH)

    knowledge_base = []

    doc_id = 1

    # ======================================================
    # Glossary
    # ======================================================

    for item in glossary:

        knowledge_base.append(

            create_document(

                doc_id=doc_id,

                doc_type="glossary",

                title=item.get("table", ""),

                content=item.get("description", ""),

                metadata={

                    "category": item.get("category"),

                    "columns": item.get("columns", [])

                }

            )

        )

        doc_id += 1

    # ======================================================
    # KPIs
    # ======================================================

    for item in kpis:

        knowledge_base.append(

            create_document(

                doc_id=doc_id,

                doc_type="kpi",

                title=item.get("name", ""),

                content=item.get("description", ""),

                metadata={

                    "formula": item.get("formula"),

                    "table": item.get("table")

                }

            )

        )

        doc_id += 1

    # ======================================================
# Business Rules
# ======================================================

    for item in rules:

        knowledge_base.append(

            create_document(

                doc_id=doc_id,

                doc_type="business_rule",

                title=item.get("title", ""),

                content=item.get("description", ""),

                metadata={

                    "tables": item.get("tables", []),

                    "columns": item.get("columns", []),

                    "related_kpis": item.get("related_kpis", []),

                    "keywords": item.get("keywords", [])

            }

        )

    )

        doc_id += 1

    # ======================================================
    # Bird Evidence Corpus
    # ======================================================

    for item in bird:

        knowledge_base.append(

            create_document(

                doc_id=doc_id,

                doc_type="bird_evidence",

                title=f"Evidence {item.get('id')}",

                content=item.get("text", ""),

                metadata={

                    "bird_id": item.get("id"),

                    "db_id": item.get("db_id")

                }

            )

        )

        doc_id += 1

    # ======================================================
    # Save
    # ======================================================

    save_json(knowledge_base, OUTPUT_PATH)

    print("=" * 60)
    print("Knowledge Base generated successfully")
    print("=" * 60)
    print(f"Glossary entries      : {len(glossary)}")
    print(f"KPI entries           : {len(kpis)}")
    print(f"Business rules        : {len(rules)}")
    print(f"BIRD evidence entries : {len(bird)}")
    print("-" * 60)
    print(f"Total documents       : {len(knowledge_base)}")
    print(f"Saved to              : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()