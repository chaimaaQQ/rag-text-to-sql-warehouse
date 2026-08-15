
from hybrid_retriever import HybridRetriever
class PromptBuilder:
    """
    Prompt Builder

    Responsibilities
    ----------------
    1. Retrieve relevant documents
    2. Convert documents into readable context
    3. Build a constrained prompt for the LLM
    """

    def __init__(self, retriever):
        self.retriever = retriever

    # ---------------------------------------------------------
    # Retrieve Context
    # ---------------------------------------------------------

    def retrieve_context(self, question, top_k=5):
        """
        Retrieve the most relevant documents.
        """

        return self.retriever.retrieve(
            query=question,
            top_k=top_k
        )

    # ---------------------------------------------------------
    # Format Context
    # ---------------------------------------------------------

    def format_context(self, documents):
        """
        Convert retrieved documents into plain text.
        """

        context = []

        for index, document in enumerate(
            documents,
            start=1
        ):

            text = []

            text.append(f"Document {index}")
            text.append(
                f"Type : {document.get('type', '')}"
            )
            text.append(
                f"Title : {document.get('title', '')}"
            )

            content = document.get("content")

            if content:
                text.append(
                    f"Description : {content}"
                )

            metadata = document.get(
                "metadata",
                {}
            )

            if metadata:

                text.append("Metadata:")

                for key, value in metadata.items():

                    text.append(
                        f"- {key}: {value}"
                    )

            context.append(
                "\n".join(text)
            )

        return "\n\n".join(context)

    # ---------------------------------------------------------
    # Build Prompt
    # ---------------------------------------------------------

    def build_prompt(self, question, top_k=5):
        """
        Build the complete constrained prompt.
        """

        documents = self.retrieve_context(
            question=question,
            top_k=top_k
        )

        context = self.format_context(
            documents
        )

        prompt = f"""
You are an expert SQL assistant specialized in TPC-DS.

Your task is to generate a SQL SELECT query
that answers the user's question.

IMPORTANT RULES:

1. Use ONLY tables and columns explicitly provided
   in the retrieved context below.

2. NEVER invent a table name.

3. NEVER invent a column name.

4. NEVER use tables or columns from your general knowledge.

5. Do NOT use a table such as "lineitem"
   unless it explicitly appears in the context.

6. Only SELECT queries are allowed.

7. Use the business rules and KPI definitions
   when they are provided.

8. If the required table or column is not present
   in the context, do NOT guess.

9. If the question cannot be answered using the
   provided context, respond exactly:

   ADDITIONAL_INFORMATION_REQUIRED

10. Return ONLY the SQL query when the question
    can be answered.

==================================================
RETRIEVED CONTEXT
==================================================

{context}

==================================================
USER QUESTION
==================================================

{question}

==================================================
FINAL INSTRUCTIONS
==================================================

Generate SQL using ONLY the retrieved context.

Do not invent schema information.

Return only SQL.
"""

        return prompt

    # ---------------------------------------------------------
    # Display Prompt
    # ---------------------------------------------------------

    def show_prompt(self, question, top_k=5):
        """
        Display the generated prompt.
        """

        prompt = self.build_prompt(
            question=question,
            top_k=top_k
        )

        print("\n")
        print("=" * 80)
        print("PROMPT SENT TO THE LLM")
        print("=" * 80)

        print(prompt)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("TESTING PROMPT BUILDER")
    print("=" * 60)

    print("\n[1] Initializing Hybrid Retriever...")

    retriever = HybridRetriever()
    retriever.initialize()

    print("\n[2] Initializing Prompt Builder...")

    builder = PromptBuilder(retriever)

    question = (
        "What is the total catalog sales revenue?"
    )

    print("\nQUESTION:")
    print(question)

    builder.show_prompt(
        question=question,
        top_k=5
    )


if __name__ == "__main__":
    main()