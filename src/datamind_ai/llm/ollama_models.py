from __future__ import annotations

import httpx


def list_ollama_models(base_url: str) -> list[str]:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        if response.status_code != 200:
            return []
        data = response.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except (httpx.RequestError, httpx.TimeoutException):
        return []


def resolve_ollama_model(
    requested: str,
    base_url: str,
    fallback: str | None = None,
) -> tuple[str, str | None]:
    """
    Resolve model name to one that exists locally.
    Returns (resolved_model, warning_message_or_none).
    """
    available = list_ollama_models(base_url)
    if not available:
        return requested or "llama3", None

    if not requested.strip():
        pick = fallback if fallback and fallback in available else available[0]
        if fallback and fallback not in available:
            for name in available:
                if name.split(":")[0] == fallback.split(":")[0]:
                    pick = name
                    break
        return pick, f"OLLAMA_MODEL não definido — a usar '{pick}'."

    if requested in available:
        return requested, None

    # Allow partial match (e.g. "llama3" -> "llama3:latest")
    for name in available:
        if name.split(":")[0] == requested.split(":")[0]:
            return name, f"Modelo '{requested}' não encontrado — a usar '{name}'."

    if fallback and fallback in available:
        return fallback, f"Modelo '{requested}' não encontrado — a usar fallback '{fallback}'."

    if fallback:
        for name in available:
            if name.split(":")[0] == fallback.split(":")[0]:
                return name, f"Modelo '{requested}' não encontrado — a usar '{name}'."

    first = available[0]
    return first, (
        f"Modelo '{requested}' não encontrado. Modelos instalados: {', '.join(available)}. "
        f"A usar '{first}'."
    )
