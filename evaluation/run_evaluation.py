import json
import os
import sys
import time

# Ajouter src/ au PYTHONPATH
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline import TextToSQLPipeline


# ============================================================
# CONFIGURATION
# ============================================================

QUESTIONS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evaluation_questions.json"
)


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():
    """Load evaluation questions from JSON."""

    if not os.path.exists(QUESTIONS_FILE):
        raise FileNotFoundError(
            f"Evaluation file not found: {QUESTIONS_FILE}"
        )

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "evaluation_questions.json must contain a JSON list."
        )

    return questions


# ============================================================
# CHECK RESULT
# ============================================================

def check_result(result, expected):
    """
    Check whether the generated SQL satisfies
    the expected table and columns.
    """

    if not result:
        return False

    if not result.get("valid", False):
        return False

    expected_table = expected.get(
        "expected_table"
    )

    expected_columns = expected.get(
        "expected_columns",
        []
    )

    generated_tables = result.get(
        "tables",
        []
    )

    generated_columns = result.get(
        "columns",
        []
    )

    # Check table
    if expected_table:
        if expected_table not in generated_tables:
            return False

    # Check columns
    for column in expected_columns:
        if column not in generated_columns:
            return False

    return True


# ============================================================
# RUN ONE TEST
# ============================================================

def run_test(pipeline, test_case):
    """Run one evaluation question."""

    test_id = test_case.get("id")
    question = test_case.get("question")

    print("\n")
    print("=" * 80)
    print(f"TEST {test_id}")
    print("=" * 80)

    print("\nQUESTION:")
    print(question)

    start_time = time.perf_counter()

    try:
        result = pipeline.run(question)

    except Exception as error:

        duration = time.perf_counter() - start_time

        print("\nERROR:")
        print(error)

        return {
            "id": test_id,
            "question": question,
            "success": False,
            "valid": False,
            "sql": None,
            "tables": [],
            "columns": [],
            "attempts": 0,
            "duration_seconds": duration,
            "error": str(error)
        }

    duration = time.perf_counter() - start_time

    success = check_result(
        result,
        test_case
    )

    print("\nGENERATED SQL:")

    if result.get("sql"):
        print(result["sql"])
    else:
        print("No SQL generated.")

    print("\nVALID:")
    print(result.get("valid", False))

    print("\nREASON:")
    print(result.get("reason", ""))

    print("\nTABLES:")
    print(result.get("tables", []))

    print("\nCOLUMNS:")
    print(result.get("columns", []))

    print("\nATTEMPTS:")
    print(result.get("attempts", 0))

    print("\nDURATION:")
    print(f"{duration:.2f} seconds")

    print("\nRESULT:")

    if success:
        print("PASS")
    else:
        print("FAIL")

    return {
        "id": test_id,
        "question": question,
        "success": success,
        "valid": result.get(
            "valid",
            False
        ),
        "sql": result.get(
            "sql"
        ),
        "reason": result.get(
            "reason",
            ""
        ),
        "tables": result.get(
            "tables",
            []
        ),
        "columns": result.get(
            "columns",
            []
        ),
        "attempts": result.get(
            "attempts",
            0
        ),
        "duration_seconds": duration
    }


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(results):
    """Calculate evaluation metrics."""

    total = len(results)

    if total == 0:
        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "success_rate": 0,
            "validity_rate": 0,
            "average_attempts": 0,
            "average_duration": 0
        }

    successful = sum(
        1
        for result in results
        if result["success"]
    )

    valid = sum(
        1
        for result in results
        if result["valid"]
    )

    total_attempts = sum(
        result["attempts"]
        for result in results
    )

    total_duration = sum(
        result["duration_seconds"]
        for result in results
    )

    return {
        "total": total,
        "successful": successful,
        "failed": total - successful,
        "success_rate": (
            successful / total * 100
        ),
        "validity_rate": (
            valid / total * 100
        ),
        "average_attempts": (
            total_attempts / total
        ),
        "average_duration": (
            total_duration / total
        )
    }


# ============================================================
# DISPLAY SUMMARY
# ============================================================

def display_summary(metrics):
    """Display final evaluation metrics."""

    print("\n")
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(
        f"\nTotal questions : "
        f"{metrics['total']}"
    )

    print(
        f"Successful      : "
        f"{metrics['successful']}"
    )

    print(
        f"Failed          : "
        f"{metrics['failed']}"
    )

    print(
        f"Success rate    : "
        f"{metrics['success_rate']:.2f}%"
    )

    print(
        f"Validity rate   : "
        f"{metrics['validity_rate']:.2f}%"
    )

    print(
        f"Average attempts: "
        f"{metrics['average_attempts']:.2f}"
    )

    print(
        f"Average duration: "
        f"{metrics['average_duration']:.2f} seconds"
    )

    print("=" * 80)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results, metrics):
    """Save evaluation results to JSON."""

    output_file = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "evaluation_results.json"
    )

    data = {
        "metrics": metrics,
        "results": results
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nResults saved to:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("TEXT-TO-SQL EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    print("\n[1] Loading evaluation questions...")

    questions = load_questions()

    print(
        f"Loaded {len(questions)} evaluation questions."
    )

    # --------------------------------------------------------
    # Initialize pipeline
    # --------------------------------------------------------

    print("\n[2] Initializing Text-to-SQL pipeline...")

    pipeline = TextToSQLPipeline(
        max_correction_attempts=3
    )

    print(
        "\nPipeline initialized successfully."
    )

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    print("\n[3] Running evaluation...")

    results = []

    for test_case in questions:

        result = run_test(
            pipeline,
            test_case
        )

        results.append(result)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print("\n[4] Calculating metrics...")

    metrics = calculate_metrics(
        results
    )

    display_summary(
        metrics
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n[5] Saving results...")

    save_results(
        results,
        metrics
    )

    print("\n")
    print("=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()