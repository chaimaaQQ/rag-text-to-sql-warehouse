"""
Text-to-SQL Pipeline.

Architecture:

    User Question
          |
          v
    Hybrid Retriever
          |
          v
    Prompt Builder
          |
          v
       Ollama
          |
          v
     SQL Extraction
          |
          v
     SQL Validator
          |
       valid?
       /    \
     YES     NO
      |       |
      |       v
      |   Auto Correction
      |       |
      |       v
      |   SQL Validator
      |       |
      v       v
    Validated SQL

IMPORTANT:
- Only SELECT queries are accepted.
- SQL is NEVER executed in this pipeline.
- The database is NOT accessed.
"""

from hybrid_retriever import HybridRetriever
from prompt_builder import PromptBuilder
from llm_client import LLMClient
from sql_validator import SQLValidator
from auto_correction import SQLAutoCorrector


class TextToSQLPipeline:
    """
    Main Text-to-SQL orchestration pipeline.
    """

    def __init__(self, max_correction_attempts=3):

        print("\n" + "=" * 80)
        print("INITIALIZING TEXT-TO-SQL PIPELINE")
        print("=" * 80)

      

        print("\n[1] Initializing Hybrid Retriever...")

        self.retriever = HybridRetriever()
        self.retriever.initialize()



        print("\n[2] Initializing Prompt Builder...")

        self.prompt_builder = PromptBuilder(
            retriever=self.retriever
        )



        print("\n[3] Initializing LLM Client...")

        self.llm_client = LLMClient()

    

        print("\n[4] Initializing SQL Validator...")

        self.validator = SQLValidator()

      

        print("\n[5] Initializing SQL Auto-Corrector...")

        self.auto_corrector = SQLAutoCorrector(
            llm_client=self.llm_client,
            validator=self.validator,
            max_attempts=max_correction_attempts
        )

        print("\n" + "=" * 80)
        print("PIPELINE INITIALIZED SUCCESSFULLY")
        print("=" * 80)


    def build_prompt(self, question, top_k=5):
        """
        Retrieve relevant business knowledge
        and build the LLM prompt.
        """

        return self.prompt_builder.build_prompt(
            question=question,
            top_k=top_k
        )
    def run(self, question, top_k=5):
        """
        Alias public de process().
        evaluation/run_evaluation.py appelle pipeline.run(question) ;
        cette méthode existe uniquement pour que l'API du pipeline
        corresponde à ce que le script d'évaluation attend, sans
        dupliquer la logique de process().
        """
        return self.process(question, top_k=top_k)


    def process(self, question, top_k=5):

        if not question or not question.strip():

            return {
                "success": False,
                "question": question,
                "sql": None,
                "valid": False,
                "reason": "Question is empty.",
                "tables": [],
                "columns": [],
                "attempts": 0
            }

        print("\n" + "=" * 80)
        print("QUESTION")
        print("=" * 80)

        print(question)

 

        print("\n" + "=" * 80)
        print("[1] BUILDING RAG PROMPT")
        print("=" * 80)

        prompt = self.build_prompt(
            question=question,
            top_k=top_k
        )

        print("\nPROMPT BUILT SUCCESSFULLY.")



        print("\n" + "=" * 80)
        print("[2] GENERATING SQL")
        print("=" * 80)

        llm_result = self.llm_client.generate_with_metadata(
            prompt=prompt,
            temperature=0.0
        )

        if not llm_result:

            return {
                "success": False,
                "question": question,
                "sql": None,
                "valid": False,
                "reason": "No response received from Ollama.",
                "tables": [],
                "columns": [],
                "attempts": 0
            }

        raw_response = llm_result.get(
            "response",
            ""
        )

        print("\nRAW LLM RESPONSE:")
        print(raw_response)



        sql = self.llm_client.extract_sql(
            raw_response
        )

        if not sql:

            return {
                "success": False,
                "question": question,
                "sql": None,
                "valid": False,
                "reason": "No SQL query found in LLM response.",
                "tables": [],
                "columns": [],
                "attempts": 0,
                "prompt_tokens": llm_result.get(
                    "prompt_tokens",
                    0
                ),
                "completion_tokens": llm_result.get(
                    "completion_tokens",
                    0
                ),
                "total_duration_ns": llm_result.get(
                    "total_duration_ns",
                    0
                )
            }

        print("\nSQL GENERATED:")
        print(sql)



        print("\n" + "=" * 80)
        print("[3] SQL VALIDATION")
        print("=" * 80)

        correction_result = self.auto_corrector.correct(
            sql=sql,
            context=prompt
        )



        final_sql = correction_result.get(
            "sql"
        )

        valid = correction_result.get(
            "valid",
            False
        )

        reason = correction_result.get(
            "reason",
            "Unknown validation result."
        )

        tables = correction_result.get(
            "tables",
            []
        )

        columns = correction_result.get(
            "columns",
            []
        )

        attempts = correction_result.get(
            "attempts",
            0
        )

        print("\n" + "=" * 80)

        if valid:
            print("SQL ACCEPTED")
        else:
            print("SQL REJECTED")

        print("=" * 80)

        print("\nFINAL SQL:")
        print(final_sql)

        print("\nVALID:")
        print(valid)

        print("\nREASON:")
        print(reason)

        print("\nTABLES:")
        print(tables)

        print("\nCOLUMNS:")
        print(columns)

        print("\nATTEMPTS:")
        print(attempts)

        print("\n" + "=" * 80)
        print("IMPORTANT: SQL WAS NOT EXECUTED.")
        print("=" * 80)

        return {
            "success": valid,
            "question": question,
            "sql": final_sql,
            "valid": valid,
            "reason": reason,
            "tables": tables,
            "columns": columns,
            "attempts": attempts,
            "prompt_tokens": llm_result.get(
                "prompt_tokens",
                0
            ),
            "completion_tokens": llm_result.get(
                "completion_tokens",
                0
            ),
            "total_duration_ns": llm_result.get(
                "total_duration_ns",
                0
            )
        }



def main():

    pipeline = TextToSQLPipeline(
        max_correction_attempts=3
    )


    question = (
        "What is the total catalog sales revenue?"
    )

    result = pipeline.process(
        question=question,
        top_k=5
    )



    print("\n")
    print("=" * 80)
    print("FINAL PIPELINE RESULT")
    print("=" * 80)

    print("\nQUESTION:")
    print(result["question"])

    print("\nSUCCESS:")
    print(result["success"])

    print("\nVALID:")
    print(result["valid"])

    print("\nREASON:")
    print(result["reason"])

    if result.get("sql"):
        print("\nFINAL SQL:")
        print(result["sql"])

    print("\nTABLES:")
    print(result.get("tables", []))

    print("\nCOLUMNS:")
    print(result.get("columns", []))

    print("\nATTEMPTS:")
    print(result.get("attempts", 0))

    print("\n" + "=" * 80)
    print("END OF PIPELINE")
    print("=" * 80)


if __name__ == "__main__":
    main()