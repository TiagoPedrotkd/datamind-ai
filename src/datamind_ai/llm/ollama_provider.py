from __future__ import annotations

import httpx

from datamind_ai.llm.base import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def name(self) -> str:
        return "Ollama (local)"

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            response = httpx.get(f"{self._base_url}/api/tags", timeout=3.0)
            return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError(
                f"Ollama não está disponível em {self._base_url}. "
                "Verifique se o serviço está a correr."
            )

        response = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        return LLMResponse(content=content, model=self._model, backend=self.name)
