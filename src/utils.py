import re


class SQLValidator:
    """
    Basic SQL validator.

    Checks:
    - SQL is not empty
    - Only SELECT queries are allowed
    - Dangerous SQL commands are rejected
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

    def validate(self, sql):
        """
        Validate a generated SQL query.

        Returns:
            dict: validation result
        """

        if not sql or not sql.strip():
            return {
                "valid": False,
                "reason": "SQL query is empty."
            }

        cleaned_sql = re.sub(
            r"```sql|```",
            "",
            sql,
            flags=re.IGNORECASE
        ).strip()

        if not re.match(r"^\s*SELECT\b", cleaned_sql, re.IGNORECASE):
            return {
                "valid": False,
                "reason": "Only SELECT queries are allowed."
            }

        for keyword in self.FORBIDDEN_KEYWORDS:
            if re.search(
                rf"\b{keyword}\b",
                cleaned_sql,
                re.IGNORECASE
            ):
                return {
                    "valid": False,
                    "reason": f"Forbidden SQL command detected: {keyword}"
                }

        return {
            "valid": True,
            "reason": "SQL query passed basic validation.",
            "sql": cleaned_sql
        }


def main():

    validator = SQLValidator()

    test_queries = [

        
        "SELECT SUM(ss_net_paid) FROM store_sales;",

        "DELETE FROM store_sales;",

        "DROP TABLE store_sales;",

        "UPDATE store_sales SET ss_net_paid = 0;"
    ]

    for index, query in enumerate(test_queries, start=1):

        print("\n" + "=" * 60)
        print(f"TEST {index}")
        print("=" * 60)

        print("SQL:")
        print(query)

        result = validator.validate(query)

        print("\nVALID:", result["valid"])
        print("REASON:", result["reason"])


if __name__ == "__main__":
    main()