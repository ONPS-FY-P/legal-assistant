"""
ollama_client.py

Purpose (SDD Section 5 -- Ollama Integration):
- Send a prompt to the locally running Ollama server (Llama 3.2).
- Ollama exposes a REST API on http://localhost:11434 by default --
  no API key needed since it's fully local.

Two modes:
- generate(): blocking, waits for the full response (simple, use this first)
- generate_stream(): yields tokens as they arrive (better UX later, once
  this is wired into the API -- streaming avoids the user staring at a
  blank screen while a local LLM thinks)
"""

import json
from collections.abc import Iterator

import requests

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2"


class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST, model: str = MODEL_NAME):
        self.host = host
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Send a prompt, block until the full response is ready, return the text.

        temperature=0.2 (not the default ~0.8) is intentional -- for a legal
        assistant we want low-randomness, consistent, conservative answers,
        not creative variation between identical questions.
        """
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["response"]

    def generate_stream(self, prompt: str, temperature: float = 0.2) -> Iterator[str]:
        """Stream tokens as they're generated -- for future use in the /ask endpoint."""
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": temperature},
            },
            timeout=120,
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "response" in chunk:
                yield chunk["response"]
            if chunk.get("done"):
                break

    def is_available(self) -> bool:
        """Check Ollama is running and the model is pulled, before we rely on it."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except requests.RequestException:
            return False


if __name__ == "__main__":
    client = OllamaClient()

    print("Checking Ollama availability...")
    if not client.is_available():
        print(f"ERROR: Ollama isn't reachable at {OLLAMA_HOST}, or model '{MODEL_NAME}' isn't pulled.")
        print("Run: ollama pull llama3.2   (and make sure `ollama serve` / the Ollama app is running)")
        raise SystemExit(1)

    print(f"Ollama is available with model '{MODEL_NAME}'. Sending a test prompt...\n")

    test_prompt = "In exactly one sentence, what is Article 21 of the Indian Constitution about?"
    answer = client.generate(test_prompt)
    print("Response:")
    print(answer)