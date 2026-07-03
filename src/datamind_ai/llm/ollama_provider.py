from __future__ import annotations

import httpx

from datamind_ai.llm.base import LLMProvider, LLMResponse
from datamind_ai.llm.ollama_models import list_ollama_models, resolve_ollama_model


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        fallback_model: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._requested_model = model
        self._fallback_model = fallback_model
        self._model = model
        self._resolve_warning: str | None = None
        self._refresh_model()

    @property
    def name(self) -> str:
        return "Ollama (local)"

    @property
    def model(self) -> str:
        return self._model

    @property
    def requested_model(self) -> str:
        return self._requested_model

    @property
    def resolve_warning(self) -> str | None:
        return self._resolve_warning

    def _refresh_model(self) -> None:
        resolved, warning = resolve_ollama_model(
            self._requested_model,
            self._base_url,
            self._fallback_model,
        )
        self._model = resolved
        self._resolve_warning = warning

    def is_available(self) -> bool:
        return bool(list_ollama_models(self._base_url))

    def list_models(self) -> list[str]:
        return list_ollama_models(self._base_url)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError(
                f"Ollama não está disponível em {self._base_url}. "
                "Verifique se o serviço está a correr."
            )

        self._refresh_model()

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

        if response.status_code == 404:
            available = self.list_models()
            raise RuntimeError(
                f"Modelo Ollama '{self._model}' não encontrado.\n\n"
                f"Modelos instalados: {', '.join(available) or '(nenhum)'}\n\n"
                "Solução:\n"
                f"  ollama pull {self._requested_model.split(':')[0]}\n"
                "ou edite o `.env` e defina todos os OLLAMA_MODEL_* para um modelo "
                f"que já tenha, por exemplo:\n"
                f"  OLLAMA_MODEL={available[0] if available else 'gemma2'}"
            )

        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "")
        return LLMResponse(content=content, model=self._model, backend=self.name)
