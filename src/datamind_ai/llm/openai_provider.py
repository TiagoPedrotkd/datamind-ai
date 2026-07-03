from __future__ import annotations

from openai import OpenAI

from datamind_ai.llm.base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: OpenAI | None = None

    @property
    def name(self) -> str:
        return "OpenAI"

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError(
                "OpenAI não configurado. Defina OPENAI_API_KEY no ficheiro .env."
            )

        response = self._get_client().chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        return LLMResponse(content=content, model=self._model, backend=self.name)
