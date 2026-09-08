"""
schema_indexer.py — indexation du schéma

Transforme les fichiers JSON produits par schema_parser.py (un par db_id)
en documents texte indexables, puis construit trois index :
    - BM25 (lexical)
    - Embeddings (sémantique, via sentence-transformers)
    - Hybride (combinaison pondérée des deux scores normalisés)

Deux granularités de documents sont indexées séparément :
    - au niveau table (pour Table Recall@k / Precision@k / MRR / nDCG@k)
    - au niveau colonne (pour Column Recall@k / Precision@k)

Dépendances à installer :
    pip install rank_bm25 sentence-transformers numpy

Usage:
    # Construire les index à partir des schémas
    python schema_indexer.py build --schemas_dir data/schemas --output_dir data/index

    # Tester une requête (mode debug rapide)
    python schema_indexer.py query --index_dir data/index --db_id california_schools \
        --question "Quel est le taux de reussite moyen par district ?" --k 5 --method hybrid
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# 1. Construction des documents texte à partir des schémas JSON
# --------------------------------------------------------------------------

def load_schemas(schemas_dir: str) -> list[dict]:
    schemas_dir = Path(schemas_dir)
    schemas = []
    for path in sorted(schemas_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            schemas.append(json.load(f))
    return schemas


def build_table_documents(schemas: list[dict]) -> list[dict]:
    """Un document par table : nom + noms de colonnes concaténés."""
    docs = []
    for schema in schemas:
        db_id = schema["db_id"]
        for table in schema["tables"]:
            col_names = ", ".join(c["name"] for c in table["columns"])
            relations = [
                f"{fk['from']} -> {fk['to']}"
                for fk in schema.get("foreign_keys", [])
                if fk.get("from", "").split(".", 1)[0] == table["name"]
            ]
            relation_text = f" Relations: {'; '.join(relations)}." if relations else ""
            text = f"Table {table['name']}. Colonnes: {col_names}.{relation_text}"
            docs.append(
                {
                    "doc_id": f"{db_id}::{table['name']}",
                    "db_id": db_id,
                    "table_name": table["name"],
                    "text": text,
                }
            )
    return docs


def build_column_documents(schemas: list[dict]) -> list[dict]:
    """Un document par colonne : table.colonne + type."""
    docs = []
    for schema in schemas:
        db_id = schema["db_id"]
        for table in schema["tables"]:
            for col in table["columns"]:
                text = f"Table {table['name']}, colonne {col['name']} (type {col['type']})."
                docs.append(
                    {
                        "doc_id": f"{db_id}::{table['name']}::{col['name']}",
                        "db_id": db_id,
                        "table_name": table["name"],
                        "column_name": col["name"],
                        "text": text,
                    }
                )
    return docs


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


# --------------------------------------------------------------------------
# 2. Construction des index
# --------------------------------------------------------------------------

def build_bm25_index(docs: list[dict]):
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(d["text"]) for d in docs]
    return BM25Okapi(tokenized_corpus)


def build_embedding_index(docs: list[dict], model_name: str = "all-MiniLM-L6-v2"):
    model = _get_cached_model(model_name)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings, model_name


def save_index(output_dir: Path, name: str, docs: list[dict], bm25, embeddings: np.ndarray, model_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{name}_docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    with open(output_dir / f"{name}_bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    np.save(output_dir / f"{name}_embeddings.npy", embeddings)
    with open(output_dir / f"{name}_model.txt", "w", encoding="utf-8") as f:
        f.write(model_name)


def build_all(schemas_dir: str, output_dir: str):
    schemas = load_schemas(schemas_dir)
    if not schemas:
        print(f"Aucun schéma trouvé dans {schemas_dir}")
        return

    output_dir = Path(output_dir)

    for level, builder in (("table", build_table_documents), ("column", build_column_documents)):
        docs = builder(schemas)
        print(f"[{level}] {len(docs)} documents à indexer")

        bm25 = build_bm25_index(docs)
        embeddings, model_name = build_embedding_index(docs)

        save_index(output_dir, level, docs, bm25, embeddings, model_name)
        print(f"[{level}] index sauvegardé dans {output_dir}")


# --------------------------------------------------------------------------
# 3. Chargement des index + retrieval (BM25 / embeddings / hybride)
# --------------------------------------------------------------------------

def load_index(index_dir: str, level: str):
    index_dir = Path(index_dir)
    with open(index_dir / f"{level}_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    with open(index_dir / f"{level}_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    embeddings = np.load(index_dir / f"{level}_embeddings.npy")
    with open(index_dir / f"{level}_model.txt", "r", encoding="utf-8") as f:
        model_name = f.read().strip()
    return docs, bm25, embeddings, model_name


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


# ----------------------------------------------------------------------
# CORRECTIF PERFORMANCE : le modèle SentenceTransformer est mis en cache
# au niveau du module et chargé UNE SEULE FOIS par nom de modèle, au lieu
# d'être ré-instancié à chaque appel de retrieve() (ce qui, sur une
# ablation de plusieurs centaines de questions, revenait à recharger le
# modèle plusieurs milliers de fois).
# ----------------------------------------------------------------------
_MODEL_CACHE: dict = {}


def _get_cached_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        print(f"[retriever_schema] Chargement du modèle {model_name} (une seule fois)...")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def retrieve(query: str, docs: list[dict], bm25, embeddings: np.ndarray, model_name: str,
             db_id: str | None = None, k: int = 5, method: str = "hybrid", alpha: float = 0.5) -> list[dict]:
    """method: 'bm25', 'embeddings' ou 'hybrid'."""

    # Filtre optionnel par db_id (recherche restreinte à la bonne base)
    candidate_idx = [i for i, d in enumerate(docs) if db_id is None or d["db_id"] == db_id]

    # BM25 : calculé seulement si nécessaire
    bm25_scores = np.array(bm25.get_scores(tokenize(query))) if method in ("bm25", "hybrid") else None

    # Embeddings : modèle mis en cache, calculé seulement si nécessaire
    emb_scores = None
    if method in ("embeddings", "hybrid"):
        model = _get_cached_model(model_name)
        query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        emb_scores = embeddings @ query_emb

    if method == "bm25":
        final_scores = bm25_scores
    elif method == "embeddings":
        final_scores = emb_scores
    elif method == "hybrid":
        final_scores = alpha * _normalize(bm25_scores) + (1 - alpha) * _normalize(emb_scores)
    else:
        raise ValueError(f"method inconnu : {method}")

    ranked = sorted(candidate_idx, key=lambda i: final_scores[i], reverse=True)[:k]
    return [{"doc": docs[i], "score": float(final_scores[i])} for i in ranked]


# --------------------------------------------------------------------------
# 4. CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Indexation et retrieval de schéma (BM25 / embeddings / hybride)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build", help="Construit les index à partir des schémas JSON")
    build_p.add_argument("--schemas_dir", required=True)
    build_p.add_argument("--output_dir", required=True)

    query_p = subparsers.add_parser("query", help="Teste une requête sur un index déjà construit")
    query_p.add_argument("--index_dir", required=True)
    query_p.add_argument("--question", required=True)
    query_p.add_argument("--db_id", default=None)
    query_p.add_argument("--k", type=int, default=5)
    query_p.add_argument("--method", choices=["bm25", "embeddings", "hybrid"], default="hybrid")
    query_p.add_argument("--level", choices=["table", "column"], default="table")

    args = parser.parse_args()

    if args.command == "build":
        build_all(args.schemas_dir, args.output_dir)
    elif args.command == "query":
        docs, bm25, embeddings, model_name = load_index(args.index_dir, args.level)
        results = retrieve(args.question, docs, bm25, embeddings, model_name,
                            db_id=args.db_id, k=args.k, method=args.method)
        for r in results:
            print(f"{r['score']:.4f}  {r['doc']['doc_id']}")


if __name__ == "__main__":
    main()
