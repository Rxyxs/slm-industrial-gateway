"""Endpoints HTTP compatibles con el contrato de OpenAI para el SLM local.

Expone `/v1/chat/completions` y `/v1/models` con el mismo contrato que la API
de OpenAI, respaldados por el motor de inferencia local (`src.engine.LLMServer`).
El endpoint de completions delega el procesamiento a `AgentOrchestrator`
(`src.agents`), que encadena:

    RouterAgent (SLM + decide tool vs. respuesta final)
    -> AnalyticsAgent (ejecuta la tool, si aplica)
    -> VerifierAgent (fidelidad + formato de la respuesta final)

Incluye métricas de Prometheus para latencia por token y throughput.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.agents import AgentMessage, AgentOrchestrator
from src.engine import GenerationConfig, GenerationError, LLMServer, ModelLoadError
from src.guardrails import GuardrailError
from src.tools import ToolExecutionError, ToolNotFoundError, ToolRegistry, build_default_registry

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


def get_orchestrator() -> AgentOrchestrator:
    """Construye el orquestador para esta solicitud.

    Deliberadamente no cacheado como singleton (a diferencia de
    `get_llm_server`/`get_tool_registry`, que sí lo son): es una envoltura
    liviana sin I/O propio, y construirlo en cada solicitud a partir de los
    singletons reales garantiza que un mock de `get_llm_server` (como en los
    tests) se refleje en el orquestador sin depender del orden de las pruebas.
    """
    return AgentOrchestrator(get_llm_server(), get_tool_registry())


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


def _count_tokens(text: str) -> int:
    """Aproxima el conteo de tokens por palabras.

    El backend GGUF (llama.cpp) no expone su tokenizer a través de
    `LLMServer.generate`, que solo devuelve el texto final.
    """
    return max(len(text.split()), 1)


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
    agent_messages = [AgentMessage(role=message.role, content=message.content) for message in request.messages]

    start = time.perf_counter()
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.run(agent_messages, config)
        completion_text = result.text
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
