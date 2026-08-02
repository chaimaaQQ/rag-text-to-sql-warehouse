"""
SQL Generator (Module 5)

Responsibilities
-----------------
1. Take a natural language question
2. Build the prompt via PromptBuilder (schema + business context)
3. Call the LLM via LLMClient to generate SQL
4. Clean / extract the raw SQL from the LLM response
5. Return a structured, traceable result (GenerationResult)
6. Optionally repeat generation N times (variance study, section 4.5)

This module is a "brick": it generates ONE SQL query per call.
The repetition logic (3 essais) lives here as a convenience method,
but the statistical aggregation of variance (mean +/- std) is the
responsibility of evaluator.py, applied AFTER SQL execution.
"""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from prompt_builder import PromptBuilder
from llm_client import LLMClient


# ============================================================
# Result structure
# ============================================================

@dataclass
class GenerationResult:
    """
    Structured, traceable output of one SQL generation call.
    Fields are aligned with the cost/latency metrics of section 11
    and the reproducibility requirements of section 16.
    """

    question: str
    sql: str                     # cleaned SQL, ready for execution
    raw_response: str            # untouched LLM output, for debugging/audit
    prompt_used: str
    model: str
    temperature: float
    seed: int | None
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    run_index: int = 0           # 0, 1, 2 for the 3 independent trials
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error: str | None = None

    @property
    def is_valid(self):
        """True if a non-empty SQL string was extracted."""
        return bool(self.sql) and self.error is None


# ============================================================
# SQL Extraction helper
# ============================================================

def extract_sql(raw_text):
    """
    Extract a clean SQL query from a raw LLM response.

    Handles three common cases:
    - SQL wrapped in a ```sql ... ``` code fence
    - SQL wrapped in a plain ``` ... ``` code fence
    - SQL surrounded by explanatory text (no fence at all)
    """

    if not raw_text:
        return ""

    # Case 1 & 2 : fenced code block
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        if candidate:
            return candidate.rstrip(";").strip() + ";"

    # Case 3 : no fence, locate the first SQL keyword and take the
    # rest of the text from there (LLMs sometimes add a preamble like
    # "Here is the SQL query:")
    match = re.search(
        r"(SELECT|WITH|INSERT|UPDATE|DELETE)\b",
        raw_text,
        re.IGNORECASE
    )
    if match:
        candidate = raw_text[match.start():].strip()
        # Stop at the first blank line followed by prose, if any
        candidate = candidate.split("\n\n")[0].strip()
        return candidate.rstrip(";").strip() + ";"

    # Nothing recognizable was found
    return ""


# ============================================================
# SQL Generator
# ============================================================

class SQLGenerator:
    """
    Orchestrates PromptBuilder + LLMClient to produce SQL queries.
    """

    def __init__(self, model="qwen2.5:7b", temperature=0.0, top_k=5):
        """
        model / temperature are fixed once and documented here,
        per section 4.5 (frozen LLM + hyperparameters for the main study).
        """

        self.builder = PromptBuilder()
        self.client = LLMClient(model=model)
        self.temperature = temperature
        self.top_k = top_k
        self.model = model

    # ---------------------------------------------------------
    # Single generation
    # ---------------------------------------------------------

    def generate_sql(self, question, run_index=0, seed=None, top_k=None):
        """
        Generate a single SQL query for one question (one trial).
        """

        top_k = top_k or self.top_k

        prompt = self.builder.build_prompt(question=question, top_k=top_k)

        start = time.perf_counter()
        result = self.client.generate_with_metadata(
            prompt,
            temperature=self.temperature,
            seed=seed
        )
        latency = time.perf_counter() - start

        if result is None:
            return GenerationResult(
                question=question,
                sql="",
                raw_response="",
                prompt_used=prompt,
                model=self.model,
                temperature=self.temperature,
                seed=seed,
                latency_seconds=latency,
                prompt_tokens=0,
                completion_tokens=0,
                run_index=run_index,
                error="LLM call failed (see llm_client logs)."
            )

        raw_response = result["response"]
        sql = extract_sql(raw_response)

        return GenerationResult(
            question=question,
            sql=sql,
            raw_response=raw_response,
            prompt_used=prompt,
            model=self.model,
            temperature=self.temperature,
            seed=seed,
            latency_seconds=latency,
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            run_index=run_index,
            error=None if sql else "No SQL could be extracted from the response."
        )

    # ---------------------------------------------------------
    # Repeated generation (variance study, section 4.5)
    # ---------------------------------------------------------

    def generate_sql_with_variance(self, question, n_runs=3, top_k=None):
        """
        Run generate_sql n_runs times independently (different seeds).
        Returns a list of GenerationResult, one per run.

        Aggregation (mean +/- std of execution accuracy, etc.) is done
        downstream in evaluator.py, after each SQL has been executed.
        """

        results = []

        for i in range(n_runs):
            seed = i  # simple, reproducible seed per run
            result = self.generate_sql(
                question=question,
                run_index=i,
                seed=seed,
                top_k=top_k
            )
            results.append(result)

        return results

    # ---------------------------------------------------------
    # Debug helper
    # ---------------------------------------------------------

    def show_generation(self, question, top_k=None):
        """
        Generate and print a single result, for quick manual testing.
        """

        result = self.generate_sql(question=question, top_k=top_k)

        print("\n" + "=" * 70)
        print("QUESTION:", result.question)
        print("=" * 70)
        print("SQL:\n", result.sql)
        print("-" * 70)
        print(f"model={result.model} | temperature={result.temperature} "
              f"| latency={result.latency_seconds:.2f}s "
              f"| tokens(in/out)={result.prompt_tokens}/{result.completion_tokens}")
        if result.error:
            print("ERROR:", result.error)
        print("=" * 70)


# ============================================================
# Main (manual test)
# ============================================================

def main():

    generator = SQLGenerator(model="qwen2.5:7b", temperature=0.0)

    question = "What is the total revenue generated by store sales?"

    generator.show_generation(question=question)


if __name__ == "__main__":
    main()