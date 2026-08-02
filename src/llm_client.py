import requests


class LLMClient:
    """
    Client responsible for communicating with Ollama.

    Responsibilities:
        - Connect to Ollama
        - Send prompts
        - Receive LLM responses
    """

    def __init__(
        self,
        model="qwen2.5:7b",
        url="http://localhost:11434/api/generate"
    ):
        self.model = model
        self.url = url

    # ---------------------------------------------------------
    # Generate (simple, text only)
    # ---------------------------------------------------------

    def generate(self, prompt, temperature=0.0, seed=None):
        """
        Send a prompt to Ollama and return the generated response (text only).
        Kept for backward compatibility with existing code.
        """

        result = self.generate_with_metadata(
            prompt,
            temperature=temperature,
            seed=seed
        )

        return result["response"] if result else None

    # ---------------------------------------------------------
    # Generate with metadata (tokens, duration)
    # ---------------------------------------------------------

    def generate_with_metadata(self, prompt, temperature=0.0, seed=None):
        """
        Same as generate(), but also returns Ollama metadata
        (tokens, duration) needed for the cost/latency metrics (section 11).
        """

        options = {"temperature": temperature}

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
                "response": data.get("response", "").strip(),
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_duration_ns": data.get("total_duration", 0),
            }

        except requests.exceptions.ConnectionError:

            print("=" * 60)
            print("Unable to connect to Ollama.")
            print("Make sure Ollama is running.")
            print("=" * 60)

            return None

        except requests.exceptions.Timeout:

            print("=" * 60)
            print("Request timed out.")
            print("=" * 60)

            return None

        except requests.exceptions.RequestException as error:

            print("=" * 60)
            print("HTTP Error")
            print(error)
            print("=" * 60)

            return None

    # ---------------------------------------------------------
    # Display Response
    # ---------------------------------------------------------

    def show_response(self, prompt):
        """
        Generate and display the LLM response.
        """

        response = self.generate(prompt)

        print("\n")
        print("=" * 60)
        print("LLM RESPONSE")
        print("=" * 60)

        if response:
            print(response)
        else:
            print("No response received.")

        print("=" * 60)


# =============================================================
# Main
# =============================================================

def main():

    client = LLMClient()

    prompt = """
You are an SQL expert.

Generate an SQL query to compute the total revenue generated
from store sales.

Return only SQL.
"""

    client.show_response(prompt)


if __name__ == "__main__":
    main()