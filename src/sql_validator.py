import json
import re
from pathlib import Path

import sqlglot
from sqlglot import exp


class SQLValidator:
    """
    SQL validator based on the TPC-DS glossary.

    Checks:
    - Only SELECT queries are allowed
    - SQL syntax is valid
    - Tables exist in the glossary
    - Columns exist in the corresponding tables
    """

    FORBIDDEN_KEYWORDS = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
    ]

    def __init__(
        self,
        glossary_path="data/business_docs/glossary.json"
    ):
        self.glossary_path = Path(glossary_path)

        self.tables = set()
        self.columns = {}

        self._load_glossary()

    # ---------------------------------------------------------
    # Load TPC-DS glossary
    # ---------------------------------------------------------

    def _load_glossary(self):
        """
        Load tables and columns from the TPC-DS glossary.
        """

        with open(
            self.glossary_path,
            "r",
            encoding="utf-8"
        ) as file:

            glossary = json.load(file)

        for document in glossary:

            table_name = document.get("table")

            if not table_name:
                continue

            table_name = table_name.lower()

            self.tables.add(table_name)

            columns = document.get("columns", [])

            self.columns[table_name] = {
                column.lower()
                for column in columns
            }

        print("=" * 60)
        print("TPC-DS GLOSSARY LOADED")
        print("=" * 60)
        print("Tables:", len(self.tables))
        print("Tables with columns:", len(self.columns))

    # ---------------------------------------------------------
    # Validate SQL
    # ---------------------------------------------------------

    def validate(self, sql):
        """
        Validate a generated SQL query.

        Returns a dictionary containing:
        - valid
        - reason
        - sql
        - tables
        - columns
        """

        # -----------------------------------------------------
        # Empty SQL
        # -----------------------------------------------------

        if not sql or not sql.strip():

            return {
                "valid": False,
                "reason": "SQL query is empty."
            }

        # -----------------------------------------------------
        # Remove Markdown code fences
        # -----------------------------------------------------

        cleaned_sql = re.sub(
            r"```sql|```",
            "",
            sql,
            flags=re.IGNORECASE
        ).strip()

        # -----------------------------------------------------
        # Only SELECT queries are allowed
        # -----------------------------------------------------

        if not re.match(
            r"^SELECT\b",
            cleaned_sql,
            re.IGNORECASE
        ):

            return {
                "valid": False,
                "reason": "Only SELECT queries are allowed."
            }

        # -----------------------------------------------------
        # Forbidden SQL commands
        # -----------------------------------------------------

        for keyword in self.FORBIDDEN_KEYWORDS:

            if re.search(
                rf"\b{keyword}\b",
                cleaned_sql,
                re.IGNORECASE
            ):

                return {
                    "valid": False,
                    "reason": (
                        f"Forbidden SQL command detected: {keyword}"
                    )
                }

        # -----------------------------------------------------
        # Parse SQL
        # -----------------------------------------------------

        try:

            parsed = sqlglot.parse_one(
                cleaned_sql,
                dialect="postgres"
            )

        except Exception as error:

            return {
                "valid": False,
                "reason": f"Invalid SQL syntax: {error}"
            }

        # -----------------------------------------------------
        # Validate tables
        # -----------------------------------------------------

        sql_tables = []

        for table in parsed.find_all(exp.Table):

            table_name = table.name.lower()

            if table_name not in sql_tables:
                sql_tables.append(table_name)

            if table_name not in self.tables:

                return {
                    "valid": False,
                    "reason": f"Unknown table: {table_name}"
                }

        # -----------------------------------------------------
        # Validate columns
        # -----------------------------------------------------

        sql_columns = []

        for column in parsed.find_all(exp.Column):

            column_name = column.name.lower()

            if column_name not in sql_columns:
                sql_columns.append(column_name)

            # -------------------------------------------------
            # Case 1: table.column
            # -------------------------------------------------

            table_name = column.table

            if table_name:

                table_name = table_name.lower()

                if table_name not in self.tables:

                    return {
                        "valid": False,
                        "reason": (
                            f"Unknown table for column: {table_name}"
                        )
                    }

                if column_name not in self.columns.get(
                    table_name,
                    set()
                ):

                    return {
                        "valid": False,
                        "reason": (
                            f"Unknown column '{column_name}' "
                            f"in table '{table_name}'"
                        )
                    }

            # -------------------------------------------------
            # Case 2: column without table qualification
            # Example:
            #
            # SELECT ss_net_paid
            # FROM store_sales;
            # -------------------------------------------------

            else:

                found = False

                for table_name in sql_tables:

                    table_columns = self.columns.get(
                        table_name,
                        set()
                    )

                    if column_name in table_columns:

                        found = True
                        break

                if not found:

                    return {
                        "valid": False,
                        "reason": (
                            f"Unknown column '{column_name}' "
                            f"in the query tables"
                        )
                    }

        # -----------------------------------------------------
        # Validation successful
        # -----------------------------------------------------

        return {
            "valid": True,
            "reason": "SQL passed validation.",
            "sql": cleaned_sql,
            "tables": sql_tables,
            "columns": sql_columns
        }


# ============================================================
# TEST
# ============================================================

def main():

    validator = SQLValidator()

    tests = [

        # -----------------------------------------------------
        # VALID
        # -----------------------------------------------------

        "SELECT SUM(ss_net_paid) FROM store_sales;",

        # -----------------------------------------------------
        # INVALID COLUMN
        # -----------------------------------------------------

        "SELECT SUM(price) FROM store_sales;",

        # -----------------------------------------------------
        # INVALID TABLE
        # -----------------------------------------------------

        "SELECT SUM(ss_net_paid) FROM unknown_table;",

        # -----------------------------------------------------
        # FORBIDDEN COMMAND
        # -----------------------------------------------------

        "DELETE FROM store_sales;"
    ]

    for index, query in enumerate(tests, start=1):

        print("\n" + "=" * 60)
        print(f"TEST {index}")
        print("=" * 60)

        print("SQL:")
        print(query)

        result = validator.validate(query)

        print("\nVALID:", result["valid"])
        print("REASON:", result["reason"])

        if result["valid"]:

            print("TABLES:", result["tables"])
            print("COLUMNS:", result["columns"])


if __name__ == "__main__":
    main()