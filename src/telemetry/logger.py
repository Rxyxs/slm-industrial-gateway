"""Logging estructurado en JSON y spans de latencia por etapa (tokenizer,
model_inference, post_processing, ...) para el pipeline multi-agente.

Distinto del `JSONLogFormatter` de `src/api/routes.py` (que loguea
inicio/fin de una solicitud HTTP completa, con un `request_id`): este
módulo mide cada ETAPA dentro de esa solicitud -- una invocación puede
disparar varios eventos (tokenizer, model_inference, post_processing, y
cualquier otra etapa de agente instrumentada), todos correlacionados por el
mismo `trace_id`, cada uno con su propio `span_id`.

Uso típico:

    from src.telemetry import start_trace, trace_stage

    start_trace()  # una vez por solicitud (ej. en el middleware HTTP)
    with trace_stage("model_inference", agent_name="engine") as span:
        texto = llm.generate(prompt)
        span.token_count = len(texto.split())

`trace_stage` también sirve como decorador (`@trace_stage(...)`) para
envolver una función entera sin abrir un bloque `with` -- es el mismo
objeto que devuelve `contextlib.contextmanager`, que soporta ambos usos de
fábrica. La diferencia práctica: como decorador no hay `span` accesible
adentro de la función (cada llamada abre su propio span), así que sirve
para latencia/errores pero no para `token_count`; para eso hace falta la
forma `with ... as span`.
"""
from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "slm_gateway_trace_id", default=None
)

# No es una lista cerrada -- cualquier string sirve como stage/agent_name --
# pero documenta el vocabulario que usa la instrumentación real del gateway
# (src/engine/llm_server.py, src/agents/orchestrator.py).
STAGE_TOKENIZER = "tokenizer"
STAGE_MODEL_INFERENCE = "model_inference"
STAGE_POST_PROCESSING = "post_processing"

_STRUCTURED_FIELDS = (
    "trace_id", "span_id", "agent_name", "stage", "latency_ms", "token_count", "status",
)


class StageJSONFormatter(logging.Formatter):
    """Un renglón JSON por evento de span: timestamp ISO 8601 UTC + los
    campos estructurados que `trace_stage` deja en `record` vía `extra`.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        for field in _STRUCTURED_FIELDS:
            valor = getattr(record, field, None)
            if valor is not None:
                payload[field] = valor
        payload["message"] = record.getMessage()
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger("slm_gateway.telemetry")
    if not logger.handlers:  # evita duplicar el handler si el módulo se reimporta (tests)
        handler = logging.StreamHandler()
        handler.setFormatter(StageJSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = _configure_logger()


def new_trace_id() -> str:
    return uuid.uuid4().hex


def start_trace(trace_id: Optional[str] = None) -> str:
    """Fija el `trace_id` del contexto actual -- una vez por solicitud
    (ej. en el middleware HTTP, o al principio de `AgentOrchestrator.run`).
    Devuelve el `trace_id` efectivo (el provisto, o uno nuevo).

    Usa `contextvars` (no una variable global ni thread-local) para que
    convivan de forma segura solicitudes concurrentes bajo el loop async de
    FastAPI -- cada tarea/corrutina ve su propio `trace_id`.
    """
    trace_id = trace_id or new_trace_id()
    _trace_id_var.set(trace_id)
    return trace_id


def current_trace_id() -> str:
    """`trace_id` del contexto actual. Si nadie llamó a `start_trace` antes
    (ej. un test que invoca `trace_stage` de forma aislada), genera uno
    nuevo en el momento -- ninguna etapa queda nunca sin `trace_id`.
    """
    trace_id = _trace_id_var.get()
    if trace_id is None:
        trace_id = start_trace()
    return trace_id


@dataclass
class _Span:
    trace_id: str
    span_id: str
    stage: str
    agent_name: str
    status: str = "ok"
    token_count: Optional[int] = None


def _emit(span: _Span, latency_ms: float, exc: Optional[BaseException] = None) -> None:
    latency_ms = round(latency_ms, 3)
    extra = {
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "agent_name": span.agent_name,
        "stage": span.stage,
        "latency_ms": latency_ms,
        "token_count": span.token_count,
        "status": span.status,
    }
    if exc is not None:
        logger.error(
            f"stage_failed stage={span.stage} agent={span.agent_name}: {exc}",
            extra=extra, exc_info=exc,
        )
    else:
        logger.info(f"stage_completed stage={span.stage} agent={span.agent_name}", extra=extra)

    # Import local (no al tope del módulo) para no crear un ciclo de imports
    # con metrics.py -- cada span completado, exitoso o no, alimenta el
    # agregador en memoria que respalda get_performance_summary().
    from src.telemetry.metrics import default_aggregator

    default_aggregator.record(
        stage=span.stage, agent_name=span.agent_name, latency_ms=latency_ms,
        status=span.status, token_count=span.token_count,
    )


@contextmanager
def trace_stage(stage: str, agent_name: str, *, trace_id: Optional[str] = None) -> Iterator[_Span]:
    """Mide una etapa (`stage`) de un agente (`agent_name`): latencia exacta,
    y `status` -- "ok" si el bloque termina sin excepción, "error" si la
    lanza (se re-lanza intacta después de loguear; nunca se traga un error
    ni interrumpe el `trace_id` del resto de la solicitud), o "fallback" si
    el propio código del bloque lo asigna a mano (ej. reintento en CPU tras
    fallar la carga en GPU).

    `token_count` se deja en `None` (omitido del log) salvo que el bloque lo
    asigne explícitamente sobre el `span` que entrega el `yield`.
    """
    span = _Span(
        trace_id=trace_id or current_trace_id(),
        span_id=uuid.uuid4().hex[:12],
        stage=stage,
        agent_name=agent_name,
    )
    inicio = time.perf_counter()
    try:
        yield span
    except Exception as exc:
        span.status = "error"
        elapsed_ms = (time.perf_counter() - inicio) * 1000
        _emit(span, elapsed_ms, exc=exc)
        raise
    else:
        elapsed_ms = (time.perf_counter() - inicio) * 1000
        _emit(span, elapsed_ms)
