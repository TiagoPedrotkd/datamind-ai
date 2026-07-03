import os
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

from datamind_ai.llm.tasks import LLMTask

load_dotenv()


class LLMBackend(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    AUTO = "auto"


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key, str(default)).lower()
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    ollama_base_url: str
    ollama_model: str
    llm_backend: LLMBackend
    ollama_models: dict[str, str]
    confidential_mode: bool
    large_dataset_rows: int
    llm_sample_rows: int

    @classmethod
    def from_env(cls) -> "Settings":
        confidential = _env_bool("CONFIDENTIAL_MODE", True)
        backend_raw = os.getenv("LLM_BACKEND", "ollama" if confidential else "auto").lower()

        if confidential and backend_raw == "openai":
            backend_raw = "ollama"

        try:
            backend = LLMBackend(backend_raw)
        except ValueError:
            backend = LLMBackend.OLLAMA if confidential else LLMBackend.AUTO

        default_ollama = os.getenv("OLLAMA_MODEL", "")
        ollama_models = {}
        for task in LLMTask:
            env_key = f"OLLAMA_MODEL_{task.value.upper()}"
            ollama_models[task.value] = os.getenv(env_key, default_ollama)

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=default_ollama,
            llm_backend=backend,
            ollama_models=ollama_models,
            confidential_mode=confidential,
            large_dataset_rows=int(os.getenv("LARGE_DATASET_ROWS", "100000")),
            llm_sample_rows=int(os.getenv("LLM_SAMPLE_ROWS", "500")),
        )

    def get_ollama_model(self, task: LLMTask | None = None) -> str:
        if task is None:
            return self.ollama_model
        return self.ollama_models.get(task.value, self.ollama_model)


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".json"}
MAX_FILE_SIZE_MB = 100
