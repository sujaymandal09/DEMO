"""Swappable LLM backends behind one interface.

    generate(prompt: str) -> Iterator[str]

Each backend yields text chunks as they are generated (streaming). The
pipeline never touches backend-specific code; it only calls `generate()`.
Switching backends is a one-line env var change: LLM_BACKEND=groq|gemini|gemma_local
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterator, Optional

SYSTEM_PROMPT = (
    "You are a concise assistant. Answer in short, complete sentences "
    "suitable for spoken output."
)


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str) -> Iterator[str]:
        """Yield text chunks as they are generated."""
        raise NotImplementedError


class GroqBackend(LLMBackend):
    name = "groq"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from groq import Groq

        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def generate(self, prompt: str) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class GeminiBackend(LLMBackend):
    name = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from google import genai

        self._genai = genai
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str) -> Iterator[str]:
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=self._genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text


class OllamaBackend(LLMBackend):
    """Local Gemma (or any Ollama model) via Ollama's HTTP API."""

    name = "gemma_local"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        import ollama

        self.model = model or os.getenv("OLLAMA_MODEL", "gemma2:2b-instruct-q4_0")
        self.client = ollama.Client(
            host=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )

    def generate(self, prompt: str) -> Iterator[str]:
        stream = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content")
            if content:
                yield content


BACKENDS = {
    "groq": GroqBackend,
    "gemini": GeminiBackend,
    "gemma_local": OllamaBackend,
}


def get_backend(name: Optional[str] = None) -> LLMBackend:
    """Factory: LLM_BACKEND env var selects the implementation."""
    name = (name or os.getenv("LLM_BACKEND", "groq")).lower()
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown LLM_BACKEND {name!r}. Choose from: {sorted(BACKENDS)}"
        )
    return BACKENDS[name]()