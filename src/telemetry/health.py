"""Diagnóstico operacional del motor SLM y de los agentes del orquestador:
combina el estado del motor (dry-run de tokenización -- barato, no gasta
cómputo de inferencia real), la presencia y tasa de error reciente de cada
agente (`src.telemetry.metrics`), y los percentiles de latencia recién
medidos en un único veredicto -- `HEALTHY`, `DEGRADED`, o `UNHEALTHY`.

A diferencia de `GET /health` (`src/api/routes.py`), que solo confirma que
el proceso responde -- un liveness probe, lo que Docker usa para decidir si
reinicia el contenedor --, esto es un readiness/diagnostic check: puede
reportar `DEGRADED` sin que el proceso esté caído.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from src.telemetry.metrics import MetricsAggregator, default_aggregator

STATUS_HEALTHY = "HEALTHY"
STATUS_DEGRADED = "DEGRADED"
STATUS_UNHEALTHY = "UNHEALTHY"
_SEVERITY = {STATUS_HEALTHY: 0, STATUS_DEGRADED: 1, STATUS_UNHEALTHY: 2}

DEFAULT_LATENCY_P95_THRESHOLD_MS = 2000.0
DEFAULT_ERROR_RATE_THRESHOLD = 0.5

# Si cualquiera de estos dos falla (tasa de error alta), el veredicto GLOBAL
# baja a UNHEALTHY, no solo DEGRADED -- coincide con los dos guardrails
# bloqueantes reales del pipeline (src/agents/orchestrator.py::run):
# SafetyComplianceAgent es la última puerta antes de responder, VerifierAgent
# la única defensa contra alucinar sobre el contexto de una tool. Un fallo
# elevado en cualquiera de los dos significa que el pipeline ya no está
# cumpliendo su función de seguridad, no solo que está lento.
CRITICAL_AGENTS = frozenset({"safety_agent", "verifier_agent"})

# agent_name (el mismo que usa trace_stage en orchestrator.py) -> atributo
# real en AgentOrchestrator. Sirve para confirmar que el orquestador quedó
# bien armado, más allá de lo que diga la telemetría reciente.
EXPECTED_AGENTS = {
    "router": "router_agent",
    "analytics_agent": "analytics_agent",
    "maintenance_advisor": "maintenance_advisor",
    "verifier_agent": "verifier_agent",
    "safety_agent": "safety_agent",
    "pdm_agent": "pdm_agent",
}


@dataclass
class ComponentHealth:
    name: str
    status: str
    detail: str


def _check_engine(llm_server: Any) -> ComponentHealth:
    if llm_server is None:
        return ComponentHealth("engine", STATUS_UNHEALTHY, "modelo no cargado (llm_server=None)")
    try:
        llm_server.dry_run()
    except Exception as exc:  # noqa: BLE001 - cualquier fallo del motor es UNHEALTHY
        return ComponentHealth("engine", STATUS_UNHEALTHY, f"dry-run de tokenizacion fallo: {exc}")
    return ComponentHealth("engine", STATUS_HEALTHY, "modelo cargado, dry-run de tokenizacion OK")


def _check_agents(
    orchestrator: Any, agent_stats: dict, error_rate_threshold: float,
) -> list[ComponentHealth]:
    resultados = []
    for agent_name, attr in EXPECTED_AGENTS.items():
        if orchestrator is None:
            resultados.append(ComponentHealth(agent_name, STATUS_UNHEALTHY, "orquestador no disponible"))
            continue
        if getattr(orchestrator, attr, None) is None:
            resultados.append(
                ComponentHealth(agent_name, STATUS_UNHEALTHY, f"'{attr}' no esta inicializado")
            )
            continue

        stats = agent_stats.get(agent_name)
        if not stats or stats["total"] == 0:
            resultados.append(ComponentHealth(agent_name, STATUS_HEALTHY, "sin invocaciones recientes"))
            continue

        detalle = f"tasa de error {stats['error_rate']:.0%} sobre {stats['total']} invocaciones recientes"
        if stats["error_rate"] > error_rate_threshold:
            status = STATUS_UNHEALTHY if agent_name in CRITICAL_AGENTS else STATUS_DEGRADED
            resultados.append(ComponentHealth(agent_name, status, detalle))
        else:
            resultados.append(ComponentHealth(agent_name, STATUS_HEALTHY, detalle))
    return resultados


def _check_latency(stage_stats: dict, latency_p95_threshold_ms: float) -> list[ComponentHealth]:
    resultados = []
    for stage, stats in stage_stats.items():
        p95 = stats.get("p95_ms")
        if p95 is None:  # sin observaciones todavia -- no hay nada que evaluar
            continue
        detalle = f"p95={p95:.1f}ms sobre {stats['count']} observaciones recientes"
        if p95 > latency_p95_threshold_ms:
            resultados.append(ComponentHealth(f"latency:{stage}", STATUS_DEGRADED, detalle))
        else:
            resultados.append(ComponentHealth(f"latency:{stage}", STATUS_HEALTHY, detalle))
    return resultados


def check_system_health(
    llm_server: Any = None,
    orchestrator: Any = None,
    aggregator: Optional[MetricsAggregator] = None,
    latency_p95_threshold_ms: float = DEFAULT_LATENCY_P95_THRESHOLD_MS,
    error_rate_threshold: float = DEFAULT_ERROR_RATE_THRESHOLD,
) -> dict:
    """Veredicto agregado: `UNHEALTHY` si el motor no responde o un agente
    crítico (`CRITICAL_AGENTS`) falla por encima de `error_rate_threshold`;
    `DEGRADED` si la latencia p95 reciente de alguna etapa supera
    `latency_p95_threshold_ms`, o un agente no crítico falla por encima del
    umbral; `HEALTHY` en cualquier otro caso -- incluido el arranque en frío
    sin datos todavía, que no se confunde con degradación real.
    """
    aggregator = aggregator or default_aggregator
    summary = aggregator.get_performance_summary()

    components = [_check_engine(llm_server)]
    components += _check_agents(orchestrator, summary["agents"], error_rate_threshold)
    components += _check_latency(summary["stages"], latency_p95_threshold_ms)

    overall = max((c.status for c in components), key=lambda s: _SEVERITY[s], default=STATUS_HEALTHY)

    return {
        "status": overall,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "components": [asdict(c) for c in components],
        "performance_summary": summary,
    }
