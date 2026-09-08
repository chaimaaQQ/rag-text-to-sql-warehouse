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
        self.db_id = None 

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



class EvidenceRetriever:
    """Retriever pour le Pipeline C (RAG métier seul) sur BIRD.

    Ne fait AUCUN retrieval : renvoie directement le champ `evidence` de la
    question courante, au format attendu par PromptBuilder.format_context().
    L'attribut `.evidence` doit être renseigné avant chaque appel, comme
    `.db_id` pour SchemaRetrieverAdapter (une pipeline traite les questions
    une par une)."""

    def __init__(self):
        self.evidence = None  

    def retrieve(self, query, top_k=5):
        if not self.evidence:
            return []
        return [{
            "type": "business_evidence",
            "title": "Connaissance métier (evidence BIRD)",
            "content": self.evidence,
            "metadata": {},
        }]


class CombinedRetriever:
    """Retriever pour le Pipeline D (RAG hybride) sur BIRD.

    Combine :
      - le retrieval de schéma (SchemaRetrieverAdapter, tables/colonnes
        pertinentes, stratégie fixée par l'ablation préliminaire section 4.3)
      - la connaissance métier fournie telle quelle (evidence BIRD, comme
        imposé section 3.2 — jamais de retrieval documentaire sur BIRD)

    À renseigner avant chaque appel : `.db_id` et `.evidence`."""

    def __init__(self, schema_adapter):
        self.schema_adapter = schema_adapter
        self.evidence_adapter = EvidenceRetriever()

    @property
    def db_id(self):
        return self.schema_adapter.db_id

    @db_id.setter
    def db_id(self, value):
        self.schema_adapter.db_id = value

    @property
    def evidence(self):
        return self.evidence_adapter.evidence

    @evidence.setter
    def evidence(self, value):
        self.evidence_adapter.evidence = value

    def retrieve(self, query, top_k=5):
        schema_docs = self.schema_adapter.retrieve(query, top_k=top_k)
        evidence_docs = self.evidence_adapter.retrieve(query, top_k=top_k)
        return schema_docs + evidence_docs