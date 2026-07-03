from __future__ import annotations

from datamind_ai.config import LLMBackend, Settings
from datamind_ai.llm.base import LLMProvider
from datamind_ai.llm.ollama_provider import OllamaProvider
from datamind_ai.llm.openai_provider import OpenAIProvider
from datamind_ai.llm.tasks import LLMTask

OLLAMA_SETUP_MSG = (
    "Ollama não está disponível. Para uso local com dados confidenciais:\n"
    "1. Instale Ollama: https://ollama.com\n"
    "2. Execute: ollama pull llama3\n"
    "3. (Opcional) ollama create datamind-chat -f modelfiles/datamind-chat.Modelfile\n"
    "4. Verifique que o serviço está a correr em http://localhost:11434"
)


class ConfidentialModeError(RuntimeError):
    """OpenAI bloqueado em modo confidencial."""


def _resolve_backend(settings: Settings) -> LLMBackend:
    if settings.confidential_mode:
        return LLMBackend.OLLAMA

    if settings.llm_backend != LLMBackend.AUTO:
        return settings.llm_backend

    ollama = OllamaProvider(
        settings.ollama_base_url, settings.get_ollama_model()
    )
    if ollama.is_available():
        return LLMBackend.OLLAMA
    return LLMBackend.OPENAI


def create_provider(
    settings: Settings | None = None,
    task: LLMTask | None = None,
) -> LLMProvider:
    settings = settings or Settings.from_env()

    if settings.confidential_mode:
        model = settings.get_ollama_model(task)
        return OllamaProvider(settings.ollama_base_url, model)

    backend = _resolve_backend(settings)
    if backend == LLMBackend.OLLAMA:
        model = settings.get_ollama_model(task)
        return OllamaProvider(settings.ollama_base_url, model)

    return OpenAIProvider(settings.openai_api_key, settings.openai_model)


def create_provider_for_task(
    task: LLMTask, settings: Settings | None = None
) -> LLMProvider:
    return create_provider(settings=settings, task=task)


def detect_available_backends(settings: Settings | None = None) -> dict[str, bool]:
    settings = settings or Settings.from_env()
    ollama = OllamaProvider(
        settings.ollama_base_url, settings.get_ollama_model()
    )
    result = {"ollama": ollama.is_available(), "openai": False}
    if not settings.confidential_mode:
        openai = OpenAIProvider(settings.openai_api_key, settings.openai_model)
        result["openai"] = openai.is_available()
    return result


def get_task_models(settings: Settings | None = None) -> dict[str, str]:
    settings = settings or Settings.from_env()
    if settings.confidential_mode or _resolve_backend(settings) == LLMBackend.OLLAMA:
        return {task.value: settings.get_ollama_model(task) for task in LLMTask}
    return {task.value: settings.openai_model for task in LLMTask}


def ensure_ollama_available(settings: Settings | None = None) -> str | None:
    settings = settings or Settings.from_env()
    ollama = OllamaProvider(
        settings.ollama_base_url, settings.get_ollama_model()
    )
    if not ollama.is_available():
        return OLLAMA_SETUP_MSG
    return None
