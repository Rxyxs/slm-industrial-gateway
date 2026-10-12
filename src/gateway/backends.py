"""Clientes asíncronos de los motores de inferencia.

Dos contratos cubren los tres motores pedidos:
- Ollama (`/api/chat`): `format` recibe el JSON Schema y el muestreo se
  restringe a él.
- OpenAI-compatible (`/v1/chat/completions`): vLLM en el borde y cualquier LLM
  en la nube. Se manda `response_format={"type": "json_schema", ...}`; vLLM lo
  implementa con guided decoding (xgrammar/outlines).

ONNX Runtime / TensorRT-LLM se integran detrás de un servidor OpenAI-compatible
(p. ej. `trtllm-serve`), así que no necesitan un cliente propio.

Temperatura 0 y semilla fija: la misma telemetría debe dar la misma acción.
"""

from __future__ import annotations

import json
from typing import Protocol

import httpx

from src.gateway.config import BackendSettings


class BackendError(Exception):
    pass


class InferenceBackend(Protocol):
    name: str
    tier: str
    max_complexity: float

    async def generate(self, system: str, user: str, schema: dict, timeout_s: float) -> str: ...

    async def aclose(self) -> None: ...

    async def warmup(self) -> None: ...


class _HttpBackend:
    def __init__(self, settings: BackendSettings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.name = settings.name
        self.tier = settings.tier
        self.max_complexity = settings.max_complexity
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def warmup(self) -> None:
        """Nada por defecto: un servidor OpenAI-compatible carga el modelo al arrancar."""

    async def _post(self, path: str, body: dict, timeout_s: float) -> dict:
        try:
            resp = await self._client.post(path, json=body, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise BackendError(f"{self.name}: timeout after {timeout_s:.2f}s") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 400:
            raise BackendError(f"{self.name}: HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise BackendError(f"{self.name}: non-JSON response") from exc


class OllamaBackend(_HttpBackend):
    async def warmup(self) -> None:
        # Ollama carga el modelo en VRAM con la primera solicitud (~35 s medidos
        # en frío con qwen2.5:1.5b). Sin esto, la primera lectura de planta tras
        # un reinicio cae siempre en failsafe por deadline.
        await self._post("/api/generate", {"model": self.settings.model, "prompt": "",
                                           "keep_alive": "30m"}, timeout_s=180.0)

    async def generate(self, system: str, user: str, schema: dict, timeout_s: float) -> str:
        body = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 7, "num_predict": 160},
        }
        data = await self._post("/api/chat", body, timeout_s)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise BackendError(f"{self.name}: unexpected response shape") from exc


class OpenAICompatBackend(_HttpBackend):
    async def generate(self, system: str, user: str, schema: dict, timeout_s: float) -> str:
        body = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "seed": 7,
            "max_tokens": 160,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "ModelOutput", "schema": schema, "strict": True},
            },
        }
        data = await self._post("/v1/chat/completions", body, timeout_s)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"{self.name}: unexpected response shape") from exc


def build_backend(settings: BackendSettings) -> InferenceBackend:
    if settings.kind == "ollama":
        return OllamaBackend(settings)
    if settings.kind == "openai":
        return OpenAICompatBackend(settings)
    raise ValueError(f"unknown backend kind '{settings.kind}'")
