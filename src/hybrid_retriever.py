from retriever_docs import DocumentRetriever


class HybridRetriever:
    """
    Hybrid Retriever

    Combines:
        - BM25 lexical retrieval
        - Embedding retrieval (FAISS)

    Returns a unified ranked list of relevant documents.
    """

    def __init__(self):

        self.retriever = DocumentRetriever()


    def initialize(self):
        """
        Initialize the hybrid retriever.

        Steps:
            1. Load Knowledge Base
            2. Prepare corpus
            3. Build BM25 index
            4. Load embedding model
            5. Build FAISS index
        """

        print("=" * 60)
        print("INITIALIZING HYBRID RETRIEVER")
        print("=" * 60)

        self.retriever.load_documents()

        self.retriever.prepare_corpus()

        self.retriever.build_bm25_index()

        self.retriever.load_embedding_model()

        self.retriever.build_embedding_index()

        print("=" * 60)
        print("Hybrid Retriever initialized successfully")
        print("=" * 60)


    def retrieve_bm25(self, query, top_k=5):
        """
        Retrieve documents using BM25.
        """

        return self.retriever.retrieve_bm25(
            query=query,
            top_k=top_k
        )



    def retrieve_embedding(self, query, top_k=5):
        """
        Retrieve documents using embeddings.
        """

        return self.retriever.retrieve_embedding(
            query=query,
            top_k=top_k
        )


    def retrieve(self, query, top_k=5):
        """
        Combine BM25 and Embedding results.

        Duplicate documents are removed using their ID.
        """

        bm25_results = self.retrieve_bm25(
            query=query,
            top_k=top_k
        )

        embedding_results = self.retrieve_embedding(
            query=query,
            top_k=top_k
        )

        merged = []

        seen = set()

        for result in bm25_results:

            document = result["document"]

            document_id = document["id"]

            if document_id not in seen:

                seen.add(document_id)

                merged.append(document)

        for result in embedding_results:

            document = result["document"]

            document_id = document["id"]

            if document_id not in seen:

                seen.add(document_id)

                merged.append(document)

        return merged[:top_k]


    def show_results(self, query, top_k=5):
        """
        Display hybrid retrieval results.
        """

        results = self.retrieve(
            query=query,
            top_k=top_k
        )

        print("\n")
        print("=" * 60)
        print("HYBRID SEARCH RESULTS")
        print("=" * 60)

        print(f"\nQuery : {query}")
        print(f"Returned documents : {len(results)}\n")

        for rank, document in enumerate(results, start=1):

            print(f"Rank    : {rank}")
            print(f"ID      : {document['id']}")
            print(f"Type    : {document['type']}")
            print(f"Title   : {document['title']}")
            print(f"Content : {document['content']}")

            metadata = document.get("metadata", {})

            if metadata:
                print("Metadata:")

                for key, value in metadata.items():

                    print(f"  {key}: {value}")

            print("-" * 60)




def main():

    retriever = HybridRetriever()

    retriever.initialize()

    query = "What is the customer count?"

    retriever.show_results(
        query=query,
        top_k=5
    )


if __name__ == "__main__":
    main()