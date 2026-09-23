"""Endpoints HTTP compatibles con el contrato de OpenAI para el SLM local, más
un endpoint propio de mantenimiento predictivo.

Expone `/v1/chat/completions` y `/v1/models` con el mismo contrato que la API
de OpenAI, respaldados por el motor de inferencia local (`src.engine.LLMServer`).
El endpoint de completions delega el procesamiento a `AgentOrchestrator`
(`src.agents`), que encadena:

    RouterAgent (SLM + decide tool vs. respuesta final)
    -> AnalyticsAgent (ejecuta la tool, si aplica; incluye MaintenanceAdvisorAgent,
       advisor, sobre el resultado de calculate_rul/sensor_anomaly_check)
    -> VerifierAgent (fidelidad + formato de la respuesta final)
    -> SafetyComplianceAgent (límites físicos de diseño, bloqueante)

`/v1/pdm/diagnose` es distinto: no pasa por el SLM ni por el orquestador de
chat. Recibe series de condición estructuradas (vibración/temperatura/carga)
y devuelve un diagnóstico multi-métrica vía `PdMAgent.diagnose()` -- para un
cliente que ya sabe qué quiere diagnosticar, no para una conversación.

Incluye métricas de Prometheus para latencia por token y throughput, logging
estructurado en JSON con un `request_id` por solicitud (ver
`request_context_middleware`), y un formato de error uniforme en todo el
gateway: toda respuesta de error, venga de donde venga, es
`{"error": "<mensaje>", "type": "<NombreDeLaExcepcion>"}` -- nunca el
`{"detail": "..."}` por defecto de FastAPI ni una traza cruda de Python
(`unhandled_exception_handler` es la red de seguridad final para lo que
ningún `except` específico atrapó).

Rate limiting: no implementado en este gateway. Ver
"Configuración (variables de entorno)" más abajo en el README para el
razonamiento y la opción recomendada (proxy inverso) en vez de agregarlo acá.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.agents import AgentMessage, AgentOrchestrator, ConditionSeries, PdMAgent
from src.engine import GenerationConfig, GenerationError, LLMServer, ModelLoadError
from src.guardrails import GuardrailError
from src.tools import ToolExecutionError, ToolNotFoundError, ToolRegistry, build_default_registry

APP_NAME = "slm-openai-gateway"
APP_VERSION = "0.1.0"

MODEL_NAME = os.environ.get("MODEL_NAME", "local-slm")
MODEL_PATH = os.environ.get("MODEL_PATH", "data/models/model.gguf")
MODEL_N_CTX = int(os.environ.get("MODEL_N_CTX", "4096"))


class JSONLogFormatter(logging.Formatter):
    """Un renglón JSON por evento -- timestamp, nivel, mensaje y `request_id`
    si el log vino de una solicitud HTTP (ver `request_context_middleware`).
    Sin dependencias externas (`python-json-logger`, etc.): el formato es
    chico y no vale la pena una dependencia nueva solo para esto."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("slm_gateway")
    if not logger.handlers:  # evita duplicar el handler si el módulo se reimporta (tests)
        handler = logging.StreamHandler()
        handler.setFormatter(JSONLogFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = _configure_logging()

app = FastAPI(title=APP_NAME, version=APP_VERSION)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Asigna un `request_id` (heredado de `X-Request-ID` si el cliente ya
    mandó uno, para que un proxy/gateway upstream pueda correlacionar sus
    propios logs con los de acá), lo expone en `request.state.request_id`
    para que los handlers y `unhandled_exception_handler` lo usen, lo agrega
    a la respuesta, y deja un log de inicio/fin con la duración real."""
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    request.state.request_id = request_id
    start = time.perf_counter()

    logger.info(
        f"request_started method={request.method} path={request.url.path}",
        extra={"request_id": request_id},
    )
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Request-ID"] = request_id
    logger.info(
        f"request_finished method={request.method} path={request.url.path} "
        f"status_code={response.status_code} duration_ms={elapsed_ms:.1f}",
        extra={"request_id": request_id},
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Aplana `HTTPException(detail={"error":..., "type":...})` a esa misma
    forma en el body -- sin esto, el handler por defecto de FastAPI la anida
    una vuelta más, como `{"detail": {"error": ..., "type": ...}}`."""
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail and "type" in detail:
        content = detail
    else:
        content = {"error": str(detail), "type": "HTTPException"}
    return JSONResponse(status_code=exc.status_code, content=content, headers=dict(exc.headers or {}))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Red de seguridad final: cualquier excepción que ningún `except`
    específico haya atrapado todavía se responde como JSON limpio (nunca la
    página de traceback por defecto de Starlette) y queda logueada con el
    `request_id` para poder correlacionarla después."""
    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        f"unhandled_exception method={request.method} path={request.url.path}",
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Error interno del servidor.", "type": type(exc).__name__},
    )

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


_pdm_agent: Optional[PdMAgent] = None


def get_pdm_agent() -> PdMAgent:
    """Construye (una única vez, de forma perezosa) el agente de diagnóstico
    predictivo, sin pasar por `get_orchestrator()`/`get_llm_server()` a
    propósito: `PdMAgent.diagnose()` no invoca al SLM en ningún punto (solo
    `calculate_rul`, pura computación), así que este endpoint no debería
    forzar la carga del modelo GGUF -- un cliente que solo usa `/v1/pdm/diagnose`
    no tiene por qué pagar ese costo ni requerir que `MODEL_PATH` exista.
    `AgentOrchestrator.diagnose_asset()` delega en el mismo `PdMAgent`, para
    quien construya el orquestador directamente en vez de vía este módulo."""
    global _pdm_agent
    if _pdm_agent is None:
        _pdm_agent = PdMAgent(get_tool_registry())
    return _pdm_agent


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


class ConditionSeriesInput(BaseModel):
    metric: str
    timestamps: list[float]
    measurements: list[float]
    failure_threshold: float


class PdMDiagnoseRequest(BaseModel):
    asset_id: str
    series: list[ConditionSeriesInput]


class MetricDiagnosisResponse(BaseModel):
    metric: str
    status: str
    trend: str
    rul_estimate: Optional[float]
    confidence: float
    has_out_of_range_readings: bool
    notes: list[str]


class PdMDiagnoseResponse(BaseModel):
    asset_id: str
    overall_status: str
    overall_confidence: float
    bottleneck_metric: Optional[str]
    recommended_maintenance_window_hours: Optional[float]
    metric_diagnoses: list[MetricDiagnosisResponse]
    summary: str


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


@app.post("/v1/pdm/diagnose", response_model=PdMDiagnoseResponse)
def diagnose_asset(request: PdMDiagnoseRequest) -> PdMDiagnoseResponse:
    """Diagnóstico de salud de un activo a partir de sus series de condición
    (vibración/temperatura/carga/etc.), vía `PdMAgent.diagnose()`.

    Deliberadamente fuera del pipeline de `/v1/chat/completions`: a diferencia
    de una tool del SLM, este endpoint no espera una decisión del modelo sobre
    qué invocar -- el cliente (un sistema SCADA/historian, por ejemplo) ya
    sabe qué series de condición tiene para un activo y pide el diagnóstico
    directamente.
    """
    agent = get_pdm_agent()
    series = [ConditionSeries(**item.model_dump()) for item in request.series]

    try:
        diagnosis = agent.diagnose(request.asset_id, series)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "type": type(exc).__name__}) from exc

    return PdMDiagnoseResponse(
        asset_id=diagnosis.asset_id,
        overall_status=diagnosis.overall_status.value,
        overall_confidence=diagnosis.overall_confidence,
        bottleneck_metric=diagnosis.bottleneck_metric,
        recommended_maintenance_window_hours=diagnosis.recommended_maintenance_window_hours,
        metric_diagnoses=[
            MetricDiagnosisResponse(
                metric=d.metric,
                status=d.status.value,
                trend=d.trend,
                rul_estimate=d.rul_estimate,
                confidence=d.confidence,
                has_out_of_range_readings=d.has_out_of_range_readings,
                notes=d.notes,
            )
            for d in diagnosis.metric_diagnoses
        ],
        summary=diagnosis.summary,
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    if not request.messages:
        raise HTTPException(
            status_code=400, detail={"error": "messages no puede estar vacío", "type": "ValueError"}
        )

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
        raise HTTPException(status_code=503, detail={"error": str(exc), "type": type(exc).__name__}) from exc
    except ValueError as exc:
        COMPLETION_REQUESTS_TOTAL.labels(model=request.model, status="error").inc()
        raise HTTPException(status_code=400, detail={"error": str(exc), "type": type(exc).__name__}) from exc
    except GenerationError as exc:
        COMPLETION_REQUESTS_TOTAL.labels(model=request.model, status="error").inc()
        raise HTTPException(status_code=500, detail={"error": str(exc), "type": type(exc).__name__}) from exc
    except ToolNotFoundError as exc:
        COMPLETION_REQUESTS_TOTAL.labels(model=request.model, status="error").inc()
        raise HTTPException(status_code=400, detail={"error": str(exc), "type": type(exc).__name__}) from exc
    except (ToolExecutionError, GuardrailError) as exc:
        COMPLETION_REQUESTS_TOTAL.labels(model=request.model, status="error").inc()
        raise HTTPException(status_code=400, detail={"error": str(exc), "type": type(exc).__name__}) from exc
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
