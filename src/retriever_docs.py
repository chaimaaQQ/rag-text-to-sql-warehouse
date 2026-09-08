import json
from pathlib import Path
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

from rank_bm25 import BM25Okapi


class DocumentRetriever:
    """
    Document Retriever using BM25.

    Phase 1:
        - Load knowledge base
        - Prepare corpus

    Phase 2:
        - Build BM25 index
        - Retrieve Top-K documents
    """

    def __init__(self):

        base_dir = Path(__file__).resolve().parent.parent

        self.knowledge_base_path = (
            base_dir
            / "data"
            / "knowledge_base"
            / "knowledge_base.json"
        )

        self.documents = []
        self.corpus = []

        self.tokenized_corpus = []
        self.bm25 = None
        self.embedding_model = None
        self.document_embeddings = None
        self.faiss_index = None


    def load_documents(self):

        with open(self.knowledge_base_path, "r", encoding="utf-8") as file:
            self.documents = json.load(file)

        print("=" * 50)
        print("Knowledge Base loaded successfully")
        print("=" * 50)
        print(f"Documents : {len(self.documents)}")

    def load_embedding_model(self):
        """
        Load the SentenceTransformer model.
        """

        print("=" * 50)
        print("Loading embedding model...")
        print("=" * 50)

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Embedding model loaded successfully.")

    def build_embedding_index(self):
        """
        Build document embeddings and create the FAISS index.
        """

        if self.embedding_model is None:
            raise RuntimeError(
                "Embedding model not loaded. "
                "Call load_embedding_model() first."
            )

        print("=" * 50)
        print("Generating document embeddings...")
        print("=" * 50)

        self.document_embeddings = self.embedding_model.encode(
            self.corpus,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        print("\nEmbedding matrix created")
        print(f"Shape : {self.document_embeddings.shape}")

        dimension = self.document_embeddings.shape[1]

        self.faiss_index = faiss.IndexFlatL2(dimension)

        self.faiss_index.add(
            self.document_embeddings.astype(np.float32)
        )

        print("\nFAISS index built")
        print(f"{self.faiss_index.ntotal} vectors indexed")

    def retrieve_embedding(self, query, top_k=5):
        """
        Retrieve Top-K documents using embeddings.
        """

        if self.faiss_index is None:
            raise RuntimeError(
                "FAISS index not built."
            )

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        distances, indices = self.faiss_index.search(
            query_embedding.astype(np.float32),
            top_k
        )

        results = []

        for rank, (index, distance) in enumerate(
            zip(indices[0], distances[0]),
            start=1
        ):

            results.append(
                {
                    "rank": rank,
                    "distance": float(distance),
                    "document": self.documents[index]
                }
            )

        return results



    def prepare_corpus(self):

        self.corpus = []

        for document in self.documents:

            text = (
                document.get("title", "")
                + " "
                + document.get("content", "")
            )

            self.corpus.append(text)

        print(f"Corpus prepared : {len(self.corpus)} documents")



    def build_bm25_index(self):

        self.tokenized_corpus = [
            document.lower().split()
            for document in self.corpus
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print("=" * 50)
        print("BM25 index built successfully")
        print("=" * 50)
        print(f"Indexed documents : {len(self.tokenized_corpus)}")


    def retrieve_bm25(self, query, top_k=5):

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []

        for rank, (index, score) in enumerate(ranked[:top_k], start=1):

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "document": self.documents[index],
                }
            )

        return results


    def show_sample(self):
        """
        Display the first raw document and its prepared corpus text.
        """

        print("\nFirst document\n")
        print(self.documents[0])

        print("\nPrepared corpus\n")
        print(self.corpus[0])

    def show_embedding_results(self, query, top_k=5):
        """
        Display embedding search results.
        """

        results = self.retrieve_embedding(
            query=query,
            top_k=top_k
        )

        print("\n")
        print("=" * 50)
        print("EMBEDDING SEARCH RESULTS")
        print("=" * 50)

        print(f"\nQuery : {query}\n")

        for result in results:

            document = result["document"]

            print(f"Rank     : {result['rank']}")
            print(f"Distance : {result['distance']:.4f}")
            print(f"ID       : {document['id']}")
            print(f"Type     : {document['type']}")
            print(f"Title    : {document['title']}")
            print(f"Content  : {document['content']}")
            print("-" * 60)





def main():

    retriever = DocumentRetriever()

    retriever.load_documents()

    retriever.prepare_corpus()

    retriever.show_sample()

    retriever.build_bm25_index()

    query = "store sales revenue"

    results = retriever.retrieve_bm25(query, top_k=5)

    print("\n" + "=" * 50)
    print("BM25 SEARCH RESULTS")
    print("=" * 50)

    print(f"\nQuery : {query}\n")

    for result in results:

        doc = result["document"]

        print(f"Rank  : {result['rank']}")
        print(f"Score : {result['score']:.4f}")
        print(f"ID    : {doc['id']}")
        print(f"Type  : {doc['type']}")
        print(f"Title : {doc['title']}")
        print(f"Content : {doc['content']}")
        print("-" * 60)

   
    retriever.load_embedding_model()

    retriever.build_embedding_index()

    retriever.show_embedding_results(
        query="store sales revenue",
        top_k=5
    )


if __name__ == "__main__":
    main()