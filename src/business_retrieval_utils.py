"""
business_retrieval_utils.py — Utilitaires partagés pour le retrieval
documentaire métier (Module 4, volet B), avec le même correctif de cache
que retriever_schema.py (le modèle SentenceTransformer est chargé UNE
SEULE FOIS par processus, pas à chaque appel de retrieve()).

Utilisé par :
  - run_business_retrieval_ablation.py (corpus TPC-DS : glossary+kpi+business_rule)
  - run_bird_evidence_retrieval_ablation.py (corpus BIRD-Evidence-Corpus)
"""

import re
import numpy as np


_MODEL_CACHE: dict = {}


def _get_cached_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        print(f"[business_retrieval_utils] Chargement du modèle {model_name} (une seule fois)...")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def doc_text(doc: dict) -> str:
    """Texte indexable d'un document knowledge_base.json (title + content)."""
    return f"{doc.get('title', '')}. {doc.get('content', '')}"


def build_index(docs: list[dict], model_name: str = "all-MiniLM-L6-v2"):
    from rank_bm25 import BM25Okapi

    texts = [doc_text(d) for d in docs]
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    model = _get_cached_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    return bm25, embeddings, model_name


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def retrieve(query: str, docs: list[dict], bm25, embeddings: np.ndarray, model_name: str,
             k: int = 5, method: str = "hybrid", alpha: float = 0.5,
             exclude_indices: set[int] | None = None) -> list[dict]:
    """method: 'bm25', 'embeddings' ou 'hybrid'. exclude_indices permet de
    masquer certains documents (ex: le document 'gold' lui-même n'est PAS
    à exclure ici — au contraire on veut voir si on le retrouve ; ce
    paramètre sert plutôt à exclure des doublons ou du bruit si besoin)."""

    candidate_idx = [i for i in range(len(docs)) if not exclude_indices or i not in exclude_indices]

    bm25_scores = np.array(bm25.get_scores(tokenize(query))) if method in ("bm25", "hybrid") else None

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
    return [{"doc_id": docs[i]["id"], "score": float(final_scores[i])} for i in ranked]


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return None
    hit = len(set(retrieved_ids) & gold_ids)
    return hit / len(gold_ids)


def precision_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not retrieved_ids:
        return 0.0
    hit = len(set(retrieved_ids) & gold_ids)
    return hit / min(k, len(retrieved_ids))