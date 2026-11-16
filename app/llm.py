"""Small, provider-neutral OpenAI-compatible chat client.

The caller never supplies an endpoint or key: both come only from server
configuration. This keeps credentials out of browsers and lets the API apply
an explicit consent gate before a transcript or idea leaves the machine.
"""
from __future__ import annotations

from typing import Any

import requests

from .errors import PipelineError
from .settings import LLMTarget


def chat_completion(
    target: LLMTarget,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    timeout: int,
) -> str:
    headers = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    try:
        response = requests.post(
            target.url + "/chat/completions",
            headers=headers,
            json={"model": target.model, "messages": messages, "temperature": temperature},
            timeout=timeout,
        )
        response.raise_for_status()
        content: Any = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("пустой ответ модели")
        return content
    except Exception as error:
        place = "облачная" if target.is_cloud else "локальная"
        raise PipelineError(f"{place.capitalize()} LLM не ответила: {error}") from error
