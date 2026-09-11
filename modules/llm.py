"""LLM inference via Ollama."""

from ollama import Client


class LLM:
    def __init__(self, base_url: str, model: str, temperature: float, max_tokens: int, system_prompt: str):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.client = Client(host=base_url)
        self.history: list[dict] = []
        print(f"[LLM] Using {model} at {base_url}")

    def chat(self, user_input: str) -> str:
        """Send user input, return assistant response. Maintains conversation history."""
        self.history.append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        reply = response["message"]["content"].strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def classify(self, prompt: str) -> str:
        """Single-turn yes/no classification — no history, minimal tokens."""
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 4},
        )
        return response["message"]["content"].strip()

    def reset(self):
        """Clear conversation history to start a fresh exchange."""
        self.history = []
