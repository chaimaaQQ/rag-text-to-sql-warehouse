"""
retriever_adapters.py — Étudiant A

PromptBuilder (module de B) attend un retriever exposant une méthode
    .retrieve(query, top_k) -> list[dict]
où chaque dict a les clés : type, title, content, metadata (optionnel).

Ce fichier fournit deux adaptateurs compatibles avec cette interface :

    NullRetriever          — ne renvoie jamais rien -> prompt sans contexte
                              (Pipeline A, baseline sans RAG)
    SchemaRetrieverAdapter — enveloppe retriever_schema.py pour renvoyer
                              les tables pertinentes au format attendu
                              (Pipeline B, RAG schéma seul)
"""

import retriever_schema as rs


class NullRetriever:
    """Retriever neutre : ne renvoie jamais de document.
    Utilisé pour le Pipeline A (sans RAG) — le prompt de B contiendra
    une section RETRIEVED CONTEXT vide, ce qui correspond bien à
    'aucun contexte externe'."""

    def retrieve(self, query, top_k=5):
        return []


class SchemaRetrieverAdapter:
    """Enveloppe retriever_schema.py (BM25/embeddings/hybride, niveau table)
    pour l'exposer avec l'interface .retrieve(query, top_k) attendue par
    PromptBuilder. Un seul index est chargé une fois ; le db_id de la
    question courante doit être renseigné avant chaque appel via
    `adapter.db_id = ...` (une pipeline traite les questions une par une,
    donc il suffit de mettre à jour l'attribut avant chaque génération)."""

    def __init__(self, index_dir: str, method: str = "hybrid", level: str = "table"):
        self.docs, self.bm25, self.embeddings, self.model_name = rs.load_index(index_dir, level)
        self.method = method
        self.db_id = None  # à renseigner avant chaque appel : adapter.db_id = "california_schools"

    def retrieve(self, query, top_k=5):
        if self.db_id is None:
            raise ValueError(
                "SchemaRetrieverAdapter.db_id n'est pas renseigné. "
                "Fais `adapter.db_id = <db_id de la question>` avant d'appeler generate_sql()."
            )

        results = rs.retrieve(
            query, self.docs, self.bm25, self.embeddings, self.model_name,
            db_id=self.db_id, k=top_k, method=self.method,
        )

        # Reformatage au format attendu par PromptBuilder.format_context()
        documents = []
        for r in results:
            doc = r["doc"]
            documents.append({
                "type": "schema_table",
                "title": doc["table_name"],
                "content": doc["text"],  # ex: "Table schools. Colonnes: CDSCode, District, School."
                "metadata": {"retrieval_score": round(r["score"], 4), "db_id": self.db_id},
            })
        return documents