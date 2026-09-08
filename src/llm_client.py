import requests

from prompt_builder import PromptBuilder
from hybrid_retriever import HybridRetriever
from sql_validator import SQLValidator


class LLMClient:
    """
    Client responsible for communicating with Ollama.

    Pipeline:
        Question
            ↓
        PromptBuilder
            ↓
        HybridRetriever
            ↓
        Ollama LLM
            ↓
        SQL extraction
            ↓
        SQL Validator
            ↓
        Validated / Rejected SQL
    """

    def __init__(
        self,
        prompt_builder=None,
        model="qwen2.5:7b",
        url="http://localhost:11434/api/generate"
    ):
        self.model = model
        self.url = url

        # Prompt builder
        self.prompt_builder = prompt_builder

        # SQL validator
        self.validator = SQLValidator()

 

    def generate(
        self,
        prompt,
        temperature=0.0,
        seed=None
    ):
        """
        Generate a response from Ollama.

        Returns only the generated text.
        """

        result = self.generate_with_metadata(
            prompt=prompt,
            temperature=temperature,
            seed=seed
        )

        if result:
            return result["response"]

        return None


    def generate_with_metadata(
        self,
        prompt,
        temperature=0.0,
        seed=None
    ):
        """
        Send prompt to Ollama and return response + metadata.
        """

        options = {
            "temperature": temperature
        }

        if seed is not None:
            options["seed"] = seed

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options
        }

        try:

            response = requests.post(
                self.url,
                json=payload,
                timeout=300
            )

            response.raise_for_status()

            data = response.json()

            return {
                "response": data.get(
                    "response",
                    ""
                ).strip(),

                "prompt_tokens": data.get(
                    "prompt_eval_count",
                    0
                ),

                "completion_tokens": data.get(
                    "eval_count",
                    0
                ),

                "total_duration_ns": data.get(
                    "total_duration",
                    0
                )
            }

        except requests.exceptions.ConnectionError:

            print("=" * 80)
            print("ERROR: Unable to connect to Ollama.")
            print("Make sure Ollama is running.")
            print("=" * 80)

            return None

        except requests.exceptions.Timeout:

            print("=" * 80)
            print("ERROR: Ollama request timed out.")
            print("=" * 80)

            return None

        except requests.exceptions.RequestException as error:

            print("=" * 80)
            print("ERROR: HTTP error while communicating with Ollama.")
            print(error)
            print("=" * 80)

            return None


    def extract_sql(self, response):
        """
        Extract SQL from an LLM response.

        Supports:

            ```sql
            SELECT ...
            ```

        and plain SELECT queries.
        """

        if not response:
            return None

        response = response.strip()



        if "```sql" in response.lower():

            parts = response.split("```")

            for part in parts:

                cleaned = part.strip()

                if cleaned.lower().startswith("sql"):

                    sql = cleaned[3:].strip()

                    return sql


        if "```" in response:

            parts = response.split("```")

            for part in parts:

                cleaned = part.strip()

                if cleaned.upper().startswith("SELECT"):

                    return cleaned

      

        if response.upper().startswith("SELECT"):

            return response

        return None



    def generate_validated_sql(
        self,
        question,
        top_k=5,
        temperature=0.0,
        seed=None
    ):
        """
        Complete RAG → LLM → SQL validation pipeline.

        Returns:
            {
                "sql": ...,
                "valid": ...,
                "reason": ...,
                "tables": ...,
                "columns": ...
            }
        """

        print("\n" + "=" * 80)
        print("BUILDING PROMPT")
        print("=" * 80)

 

        if self.prompt_builder is None:

            return {
                "sql": None,
                "valid": False,
                "reason": "PromptBuilder is not initialized.",
                "tables": [],
                "columns": []
            }



        prompt = self.prompt_builder.build_prompt(
            question=question,
            top_k=top_k
        )

        print("\nPROMPT BUILT SUCCESSFULLY.")


        result = self.generate_with_metadata(
            prompt=prompt,
            temperature=temperature,
            seed=seed
        )

        if not result:

            return {
                "sql": None,
                "valid": False,
                "reason": "No response received from Ollama.",
                "tables": [],
                "columns": []
            }



        raw_response = result["response"]

        print("\n" + "=" * 80)
        print("RAW LLM RESPONSE")
        print("=" * 80)

        print(raw_response)


        sql = self.extract_sql(raw_response)

        if not sql:

            return {
                "sql": None,
                "valid": False,
                "reason": "No SQL query found in LLM response.",
                "tables": [],
                "columns": []
            }

        print("\n" + "=" * 80)
        print("SQL GENERATED")
        print("=" * 80)

        print(sql)



        validation = self.validator.validate(sql)

        print("\n" + "=" * 80)
        print("SQL VALIDATION")
        print("=" * 80)

        print(
            "VALID:",
            validation["valid"]
        )

        print(
            "REASON:",
            validation["reason"]
        )



        if validation["valid"]:

            print("\nSQL ACCEPTED.")

        else:

            print("\nSQL REJECTED.")



        return {
            "sql": sql,
            "valid": validation["valid"],
            "reason": validation["reason"],

            "tables": validation.get(
                "tables",
                []
            ),

            "columns": validation.get(
                "columns",
                []
            ),

            "prompt_tokens": result.get(
                "prompt_tokens",
                0
            ),

            "completion_tokens": result.get(
                "completion_tokens",
                0
            ),

            "total_duration_ns": result.get(
                "total_duration_ns",
                0
            )
        }



    def show_response(
        self,
        question,
        top_k=5
    ):
        """
        Run the complete RAG → LLM → Validator pipeline.
        """

        result = self.generate_validated_sql(
            question=question,
            top_k=top_k
        )

        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)

        if result.get("sql"):

            print("\nSQL:")
            print(result["sql"])

        else:

            print("\nNo SQL generated.")

        print(
            "\nVALID:",
            result["valid"]
        )

        print(
            "REASON:",
            result["reason"]
        )

        if result.get("tables"):

            print(
                "TABLES:",
                result["tables"]
            )

        if result.get("columns"):

            print(
                "COLUMNS:",
                result["columns"]
            )

        if result.get("prompt_tokens"):

            print(
                "PROMPT TOKENS:",
                result["prompt_tokens"]
            )

        if result.get("completion_tokens"):

            print(
                "COMPLETION TOKENS:",
                result["completion_tokens"]
            )

        print("=" * 80)




def main():

    print("=" * 80)
    print("TESTING LLM CLIENT")
    print("=" * 80)



    print("\n[1] Initializing Hybrid Retriever...")

    retriever = HybridRetriever()

    retriever.initialize()



    print("\n[2] Initializing Prompt Builder...")

    prompt_builder = PromptBuilder(
        retriever=retriever
    )



    print("\n[3] Initializing LLM Client...")

    client = LLMClient(
        prompt_builder=prompt_builder,
        model="qwen2.5:7b",
        url="http://localhost:11434/api/generate"
    )


    question = (
        "What is the total revenue generated "
        "by catalog sales?"
    )

    print("\nQUESTION:")
    print(question)



    client.show_response(
        question=question,
        top_k=5
    )


if __name__ == "__main__":
    main()