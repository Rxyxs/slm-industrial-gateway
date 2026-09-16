"""Endpoints HTTP compatibles con el contrato de OpenAI para el SLM local.

Expone `/v1/chat/completions` y `/v1/models` con el mismo contrato que la API
de OpenAI, respaldados por el motor de inferencia local (`src.engine.LLMServer`).
El endpoint de completions orquesta el flujo completo del agente industrial:

    validación de entrada -> inferencia SLM -> ejecución de tool (si aplica)
    -> guardrails de salida -> respuesta JSON

Incluye métricas de Prometheus para latencia por token y throughput.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from decimal import Decimal
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from src.engine import GenerationConfig, GenerationError, LLMServer, ModelLoadError
from src.guardrails import GuardrailError, OutputValidationError, validate_json_output
from src.tools import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
    build_default_registry,
)

APP_NAME = "slm-openai-gateway"
APP_VERSION = "0.1.0"

MODEL_NAME = os.environ.get("MODEL_NAME", "local-slm")
MODEL_PATH = os.environ.get("MODEL_PATH", "data/models/model.gguf")
MODEL_N_CTX = int(os.environ.get("MODEL_N_CTX", "4096"))

app = FastAPI(title=APP_NAME, version=APP_VERSION)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Métricas de negocio: el instrumentator de arriba ya cubre latencia/conteo HTTP
# genérico; estas métricas propias miden lo específico de la generación del SLM.
TOKEN_LATENCY_SECONDS = Histogram(
    "slm_token_latency_seconds",
    "Latencia de generación por token de salida (segundos/token).",
    labelnames=("model",),
)
TOKENS_GENERATED_TOTAL = Counter(
    "slm_tokens_generated_total",
    "Tokens de salida generados; permite derivar el throughput (tokens/segundo).",
    labelnames=("model",),
)
COMPLETION_REQUESTS_TOTAL = Counter(
    "slm_completion_requests_total",
    "Solicitudes de completion procesadas, por resultado.",
    labelnames=("model", "status"),
)

_llm_server: Optional[LLMServer] = None
_tool_registry: Optional[ToolRegistry] = None


def get_llm_server() -> LLMServer:
    """Carga (una única vez, de forma perezosa) el servidor de inferencia local."""
    global _llm_server
    if _llm_server is None:
        _llm_server = LLMServer(MODEL_PATH, n_ctx=MODEL_N_CTX)
    return _llm_server


def get_tool_registry() -> ToolRegistry:
    """Construye (una única vez, de forma perezosa) el registro de herramientas industriales."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = build_default_registry()
    return _tool_registry


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    stop: Optional[list[str]] = None
    stream: bool = False
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "local"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ToolCallEnvelope(BaseModel):
    """Formato estructurado en el que el SLM solicita la ejecución de una tool."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


TOOL_CALL_INSTRUCTIONS = (
    "Si necesitas una herramienta para responder, contesta ÚNICAMENTE con un objeto JSON "
    'con este formato exacto: {"tool": "<nombre_herramienta>", "arguments": {<argumentos>}}. '
    "Si no necesitas ninguna herramienta, responde directamente en lenguaje natural."
)


def _count_tokens(text: str) -> int:
    """Aproxima el conteo de tokens por palabras.

    El backend GGUF (llama.cpp) no expone su tokenizer a través de
    `LLMServer.generate`, que solo devuelve el texto final.
    """
    return max(len(text.split()), 1)


def _build_prompt(messages: list[ChatMessage], registry: ToolRegistry) -> str:
    tool_names = ", ".join(registry.list_tools())
    preamble = f"system: Herramientas disponibles: {tool_names}. {TOOL_CALL_INSTRUCTIONS}"
    turns = [preamble] + [f"{message.role}: {message.content}" for message in messages]
    turns.append("assistant:")
    return "\n".join(turns)


def _try_parse_tool_call(text: str) -> Optional[ToolCallEnvelope]:
    """Intenta interpretar `text` como una solicitud de ejecución de tool.

    Devuelve `None` (texto en lenguaje natural) si `text` no es un JSON válido
    que cumpla el esquema de `ToolCallEnvelope`.
    """
    try:
        return validate_json_output(text, ToolCallEnvelope)  # type: ignore[return-value]
    except OutputValidationError:
        return None


def _json_default(value: Any) -> Any:
    """Serializa tipos que DuckDB puede devolver y que `json` no soporta de forma nativa."""
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _apply_output_guardrails(text: str) -> str:
    """Guardrail de salida: rechaza respuestas vacías antes de devolverlas al cliente."""
    if not isinstance(text, str) or not text.strip():
        raise GenerationError("El modelo produjo una respuesta vacía o inválida.")
    return text


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id=MODEL_NAME, created=int(time.time()))])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages no puede estar vacío")

    default_max_tokens = GenerationConfig().max_tokens
    config = GenerationConfig(
        max_tokens=request.max_tokens or default_max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop or [],
    )
    registry = get_tool_registry()

    start = time.perf_counter()
    try:
        server = get_llm_server()
        completion_text = server.generate(_build_prompt(request.messages, registry), config=config)

        # Ejecución de tool (si aplica): si el SLM solicitó una herramienta en
        # lugar de responder en lenguaje natural, se despacha y se le devuelve
        # el resultado en un segundo turno para que redacte la respuesta final.
        tool_call = _try_parse_tool_call(completion_text)
        if tool_call is not None:
            tool_result = registry.dispatch(tool_call.tool, tool_call.arguments)
            follow_up_messages = [
                *request.messages,
                ChatMessage(role="assistant", content=completion_text),
                ChatMessage(
                    role="tool",
                    content=json.dumps(tool_result, ensure_ascii=False, default=_json_default),
                ),
            ]
            completion_text = server.generate(
                _build_prompt(follow_up_messages, registry), config=config
            )

        completion_text = _apply_output_guardrails(completion_text)
    except ModelLoadError as exc:
        COMPLETION_REQUESTS_TOTAL.labels(model=request.model, status="error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        COMPLETION_REQUESTS_TOTAL.labels(model=request.model, status="error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GenerationError as exc:
        COMPLETION_REQUESTS_TOTAL.labels(model=request.model, status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ToolNotFoundError as exc:
        COMPLETION_REQUESTS_TOTAL.labels(model=request.model, status="error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ToolExecutionError, GuardrailError) as exc:
        COMPLETION_REQUESTS_TOTAL.labels(model=request.model, status="error").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed = time.perf_counter() - start

    prompt_tokens = sum(_count_tokens(message.content) for message in request.messages)
    completion_tokens = _count_tokens(completion_text)

    TOKENS_GENERATED_TOTAL.labels(model=request.model).inc(completion_tokens)
    TOKEN_LATENCY_SECONDS.labels(model=request.model).observe(elapsed / completion_tokens)
    COMPLETION_REQUESTS_TOTAL.labels(model=request.model, status="success").inc()

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=completion_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
