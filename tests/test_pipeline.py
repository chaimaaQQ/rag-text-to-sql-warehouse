"""
Test suite for the Text-to-SQL RAG pipeline.

Tests:
    1. Store sales revenue
    2. Catalog sales revenue
    3. Web sales revenue
    4. Catalog returns
    5. Empty question
    6. Catalog sales revenue with different wording

IMPORTANT:
    SQL is NOT executed against a database.
    The tests verify:
        - RAG retrieval
        - Prompt construction
        - SQL generation
        - SQL validation
        - Auto-correction integration
"""

import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from pipeline import TextToSQLPipeline


# ============================================================
# TEST HELPER
# ============================================================

def run_test(
    pipeline,
    test_number,
    question,
    expected_valid=True,
    expected_table=None,
    expected_column=None
):
    """
    Execute one pipeline test.
    """

    print("\n")
    print("=" * 80)
    print(f"TEST {test_number}")
    print("=" * 80)

    print("\nQUESTION:")
    print(question)

    try:

        # IMPORTANT:
        # TextToSQLPipeline uses process(), not run().
        result = pipeline.process(
            question=question,
            top_k=5
        )

    except Exception as error:

        print("\n❌ TEST ERROR")
        print(type(error).__name__)
        print(str(error))

        return False

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\nRESULT:")
    print("SUCCESS:", result.get("success"))
    print("VALID:", result.get("valid"))
    print("REASON:", result.get("reason"))

    print("\nSQL:")

    if result.get("sql"):
        print(result["sql"])
    else:
        print("None")

    print("\nTABLES:")
    print(result.get("tables", []))

    print("\nCOLUMNS:")
    print(result.get("columns", []))

    print("\nATTEMPTS:")
    print(result.get("attempts", 0))

    # ========================================================
    # CHECK VALIDITY
    # ========================================================

    actual_valid = result.get("valid", False)

    if actual_valid != expected_valid:

        print("\n❌ FAILED")

        print(
            f"Expected VALID={expected_valid}, "
            f"but received VALID={actual_valid}"
        )

        return False

    # ========================================================
    # CHECK EXPECTED TABLE
    # ========================================================

    if expected_table is not None:

        tables = result.get("tables", [])

        if expected_table not in tables:

            print("\n❌ FAILED")

            print(
                f"Expected table: {expected_table}"
            )

            print(
                f"Returned tables: {tables}"
            )

            return False

    # ========================================================
    # CHECK EXPECTED COLUMN
    # ========================================================

    if expected_column is not None:

        columns = result.get("columns", [])

        if expected_column not in columns:

            print("\n❌ FAILED")

            print(
                f"Expected column: {expected_column}"
            )

            print(
                f"Returned columns: {columns}"
            )

            return False

    # ========================================================
    # CHECK SQL
    # ========================================================

    if expected_valid:

        sql = result.get("sql")

        if not sql:

            print("\n❌ FAILED")
            print("Expected SQL but received None.")

            return False

        if not sql.strip().upper().startswith("SELECT"):

            print("\n❌ FAILED")
            print("Generated query is not a SELECT query.")

            return False

    # ========================================================
    # SUCCESS
    # ========================================================

    print("\n✅ PASSED")

    return True


# ============================================================
# MAIN TEST SUITE
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("TEXT-TO-SQL PIPELINE TEST SUITE")
    print("=" * 80)

    # ========================================================
    # INITIALIZE PIPELINE
    # ========================================================

    print("\nInitializing pipeline...")

    try:

        pipeline = TextToSQLPipeline(
            max_correction_attempts=3
        )

    except Exception as error:

        print("\n❌ PIPELINE INITIALIZATION FAILED")

        print(
            type(error).__name__
        )

        print(
            str(error)
        )

        return

    print("\nPipeline initialized successfully.")

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    # ========================================================
    # TEST 1
    # ========================================================

    results.append(
        run_test(
            pipeline=pipeline,
            test_number=1,
            question=(
                "What is the total revenue "
                "generated by store sales?"
            ),
            expected_valid=True,
            expected_table="store_sales",
            expected_column="ss_net_paid"
        )
    )

    # ========================================================
    # TEST 2
    # ========================================================

    results.append(
        run_test(
            pipeline=pipeline,
            test_number=2,
            question=(
                "What is the total catalog sales revenue?"
            ),
            expected_valid=True,
            expected_table="catalog_sales",
            expected_column="cs_net_paid"
        )
    )

    # ========================================================
    # TEST 3
    # ========================================================

    results.append(
        run_test(
            pipeline=pipeline,
            test_number=3,
            question=(
                "What is the total web sales revenue?"
            ),
            expected_valid=True,
            expected_table="web_sales",
            expected_column="ws_net_paid"
        )
    )

    # ========================================================
    # TEST 4
    # ========================================================

    results.append(
        run_test(
            pipeline=pipeline,
            test_number=4,
            question=(
                "What is the total catalog returns amount?"
            ),
            expected_valid=True,
            expected_table="catalog_returns",
            expected_column="cr_return_amount"
        )
    )

    # ========================================================
    # TEST 5
    # ========================================================

    results.append(
        run_test(
            pipeline=pipeline,
            test_number=5,
            question="",
            expected_valid=False
        )
    )

    # ========================================================
    # TEST 6
    # ========================================================

    results.append(
        run_test(
            pipeline=pipeline,
            test_number=6,
            question=(
                "What is the total revenue "
                "generated by catalog sales?"
            ),
            expected_valid=True,
            expected_table="catalog_sales",
            expected_column="cs_net_paid"
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_tests = len(results)
    passed_tests = sum(results)
    failed_tests = total_tests - passed_tests

    print("\nTotal tests :", total_tests)
    print("Passed      :", passed_tests)
    print("Failed      :", failed_tests)

    print("\nResults:")

    for index, passed in enumerate(results, start=1):

        if passed:
            status = "PASSED"
        else:
            status = "FAILED"

        print(
            f"  Test {index}: {status}"
        )

    print("\n")

    if failed_tests == 0:

        print("=" * 80)
        print("✅ ALL PIPELINE TESTS PASSED")
        print("=" * 80)

    else:

        print("=" * 80)
        print("❌ SOME PIPELINE TESTS FAILED")
        print("=" * 80)

    print("\nIMPORTANT:")
    print("SQL was NOT executed against a database.")
    print(
        "Tests verify RAG, SQL generation, "
        "validation and correction."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()