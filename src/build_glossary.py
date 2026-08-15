import json
import csv
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TPCDS_DIR = PROJECT_ROOT / "data" / "raw" / "tpcds"
OUTPUT_DIR = PROJECT_ROOT / "business_docs"
OUTPUT_FILE = OUTPUT_DIR / "glossary.json"

FACT_TABLES = {
    "store_sales",
    "store_returns",
    "catalog_sales",
    "catalog_returns",
    "web_sales",
    "web_returns",
    "inventory",
}


def find_data_file(table_path: Path):
    """
    Recherche le fichier de données dans un dossier de table.
    """

    extensions = [
        "*.csv",
        "*.CSV",
        "*.dat",
        "*.DAT",
        "*.tbl",
        "*.parquet"
    ]

    for ext in extensions:
        files = list(table_path.glob(ext))
        if files:
            return files[0]

    return None




def extract_columns(file_path: Path):
    """
    Extrait les colonnes d'un fichier CSV ou Parquet.
    """

    if file_path is None:
        return []

    try:

        suffix = file_path.suffix.lower()

        if suffix == ".csv":

            df = pd.read_csv(file_path, nrows=0)
            return list(df.columns)

        elif suffix == ".parquet":

            df = pd.read_parquet(file_path)
            return list(df.columns)

        else:

            return []

    except Exception as e:

        print(f"Erreur lecture {file_path.name}: {e}")
        return []

def generate_description(table_name, category):

    words = table_name.replace("_", " ")

    if category == "fact_table":
        return (
            f"Fact table containing transactional records related to {words}."
        )

    return (
        f"Dimension table describing {words}."
    )


def build_entry(table_dir: Path):

    category = (
        "fact_table"
        if table_dir.name in FACT_TABLES
        else "dimension_table"
    )

    data_file = find_data_file(table_dir)

    columns = extract_columns(data_file)

    return {
        "table": table_dir.name,
        "category": category,
        "description": generate_description(
            table_dir.name,
            category
        ),
        "data_file": data_file.name if data_file else None,
        "number_of_columns": len(columns),
        "columns": columns
    }


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    glossary = []

    for table in sorted(TPCDS_DIR.iterdir()):

        if table.is_dir():

            glossary.append(build_entry(table))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(
            glossary,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 60)
    print(f"Tables trouvées : {len(glossary)}")
    print(f"Glossaire généré : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()