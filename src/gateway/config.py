"""Configuración del gateway de planta, leída una sola vez desde variables de entorno.

Todo valor que cambie entre un notebook de desarrollo y un IPC en planta vive acá
(URLs de backends, umbrales del circuit breaker, presupuesto de latencia, rutas de
auditoría). El resto del paquete recibe un `GatewaySettings` y no vuelve a mirar
`os.environ`, así los tests construyen configuraciones explícitas sin parchear el
entorno del proceso.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class BackendSettings:
    name: str
    kind: str  # "ollama" | "openai"  (vLLM y la nube hablan el contrato OpenAI)
    base_url: str
    model: str
    tier: str  # "edge" | "cloud"
    timeout_s: float = 2.0
    api_key: str = ""
    max_complexity: float = 1.0  # por encima de esto el router no lo elige


@dataclass(frozen=True)
class GatewaySettings:
    backends: list[BackendSettings] = field(default_factory=list)

    # Presupuesto total de una inferencia síncrona, incluyendo fallbacks.
    deadline_s: float = 3.0
    # Complejidad sobre la cual el router salta directo a la nube (si existe).
    cloud_complexity_threshold: float = 0.7
    # En redes OT aisladas no hay nube: el router nunca la intenta.
    allow_cloud: bool = False

    breaker_failure_threshold: int = 3
    breaker_reset_timeout_s: float = 15.0

    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 20

    cache_ttl_s: float = 30.0
    cache_max_items: int = 2048

    redis_url: str = ""
    audit_db_path: str = "data/audit/gateway_audit.duckdb"

    # Claves válidas. Se guardan como SHA-256; nunca se loguean en claro.
    api_keys: list[str] = field(default_factory=list)
    # Si es False, una recomendación validada NO se escribe al PLC: solo se
    # publica para que un operador la confirme (modo asesor, el default seguro).
    allow_plc_writeback: bool = False

    # "snapshot": solo la telemetría actual (comportamiento original).
    # "context": hechos de tendencia/laboratorio/soft sensor + procedimiento.
    # "context_fewshot": lo anterior más ejemplos resueltos.
    # "context_derived": "context" más el estado de la sílice ya calculado (silica_state).
    # "context_gated": "context_derived" y además la gramática solo ofrece ADJUST cuando el
    #   procedimiento, calculado en código, lo permite (silica_state == SUSTAINED_HIGH).
    #   Default: es la única variante que en la evaluación cumplió las tres metas a la vez.
    prompt_mode: str = "context_gated"
    context_window_h: int = 8
    soft_sensor_path: str = ""


def load_settings() -> GatewaySettings:
    backends: list[BackendSettings] = []

    if os.environ.get("EDGE_OLLAMA_URL", "http://localhost:11434"):
        backends.append(
            BackendSettings(
                name="edge-ollama",
                kind="ollama",
                base_url=os.environ.get("EDGE_OLLAMA_URL", "http://localhost:11434"),
                model=os.environ.get("EDGE_OLLAMA_MODEL", "qwen2.5:1.5b-instruct"),
                tier="edge",
                timeout_s=float(os.environ.get("EDGE_TIMEOUT_S", "2.5")),
                max_complexity=float(os.environ.get("EDGE_MAX_COMPLEXITY", "0.7")),
            )
        )
    if os.environ.get("EDGE_VLLM_URL"):
        backends.append(
            BackendSettings(
                name="edge-vllm",
                kind="openai",
                base_url=os.environ["EDGE_VLLM_URL"],
                model=os.environ.get("EDGE_VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
                tier="edge",
                timeout_s=float(os.environ.get("EDGE_TIMEOUT_S", "2.5")),
                max_complexity=float(os.environ.get("EDGE_MAX_COMPLEXITY", "0.7")),
            )
        )
    if os.environ.get("CLOUD_LLM_URL"):
        backends.append(
            BackendSettings(
                name="cloud",
                kind="openai",
                base_url=os.environ["CLOUD_LLM_URL"],
                model=os.environ.get("CLOUD_LLM_MODEL", "gpt-4o-mini"),
                tier="cloud",
                timeout_s=float(os.environ.get("CLOUD_TIMEOUT_S", "8")),
                api_key=os.environ.get("CLOUD_LLM_API_KEY", ""),
            )
        )

    return GatewaySettings(
        backends=backends,
        deadline_s=float(os.environ.get("GATEWAY_DEADLINE_S", "3.0")),
        cloud_complexity_threshold=float(os.environ.get("CLOUD_COMPLEXITY_THRESHOLD", "0.7")),
        allow_cloud=_env_bool("ALLOW_CLOUD", False),
        breaker_failure_threshold=int(os.environ.get("BREAKER_FAILURES", "3")),
        breaker_reset_timeout_s=float(os.environ.get("BREAKER_RESET_S", "15")),
        rate_limit_per_minute=int(os.environ.get("RATE_LIMIT_PER_MIN", "120")),
        rate_limit_burst=int(os.environ.get("RATE_LIMIT_BURST", "20")),
        cache_ttl_s=float(os.environ.get("CACHE_TTL_S", "30")),
        redis_url=os.environ.get("REDIS_URL", ""),
        audit_db_path=os.environ.get("AUDIT_DB_PATH", "data/audit/gateway_audit.duckdb"),
        api_keys=_env_list("GATEWAY_API_KEYS"),
        allow_plc_writeback=_env_bool("ALLOW_PLC_WRITEBACK", False),
        prompt_mode=os.environ.get("PROMPT_MODE", "context_gated"),
        context_window_h=int(os.environ.get("CONTEXT_WINDOW_H", "8")),
        soft_sensor_path=os.environ.get("SOFT_SENSOR_PATH", ""),
    )
