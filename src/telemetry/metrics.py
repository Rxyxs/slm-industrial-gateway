"""Agregador en memoria de las métricas de rendimiento del pipeline:
tokens/seg, latencias p50/p95/p99 por etapa, y tasa de errores por agente.

Se alimenta de los mismos spans reales que ya emite `trace_stage`
(`src/telemetry/logger.py`) -- cada span completado, exitoso o no, se
registra automáticamente en `default_aggregator` (ver el `import` al final
de `logger.py`), así que `get_performance_summary()` siempre refleja
invocaciones reales del pipeline, no una contabilidad paralela que alguien
tiene que acordarse de actualizar a mano.

Ventana acotada a propósito: `MetricsAggregator` es pensado para un proceso
de servidor de larga duración, así que las latencias por etapa se guardan en
un `deque(maxlen=...)` -- las observaciones más viejas se descartan solas en
vez de crecer sin límite. Los contadores (tokens totales, total/errores por
agente) sí son acumulados de por vida: son sumas escalares, no listas, así
que no tienen el mismo problema de memoria.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Literal, no importado de logger.py: evitar un ciclo de imports entre los
# dos módulos de src/telemetry/ (logger.py necesita importar este archivo
# para alimentar el agregador desde `_emit`).
_STAGE_MODEL_INFERENCE = "model_inference"

DEFAULT_WINDOW_SIZE = 1000


@dataclass
class _StageStats:
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_WINDOW_SIZE))
    tokens_total: int = 0
    inference_seconds_total: float = 0.0


@dataclass
class _AgentStats:
    total: int = 0
    errors: int = 0


def _percentile(valores, pct: float) -> Optional[float]:
    """Percentil por interpolación lineal (`numpy.percentile`, método
    estándar) sobre lo que quede en la ventana. `None` si la ventana está
    vacía -- un percentil de una muestra vacía no es 0.0, es "sin datos
    todavía", y `check_system_health` depende de esa distinción para no
    reportar DEGRADED en el arranque en frío antes de la primera solicitud.
    """
    if not valores:
        return None
    return float(np.percentile(np.asarray(valores, dtype=float), pct))


class MetricsAggregator:
    """Agregador thread-safe (protegido por un `Lock`, porque FastAPI puede
    atender solicitudes concurrentes y varios threads/tasks pueden emitir
    spans al mismo tiempo)."""

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._lock = threading.Lock()
        self._stages: dict[str, _StageStats] = {}
        self._agents: dict[str, _AgentStats] = {}

    def record(self, *, stage: str, agent_name: str, latency_ms: float,
               status: str, token_count: Optional[int] = None) -> None:
        with self._lock:
            stage_stats = self._stages.setdefault(
                stage, _StageStats(latencies_ms=deque(maxlen=self._window_size))
            )
            stage_stats.latencies_ms.append(latency_ms)
            if stage == _STAGE_MODEL_INFERENCE and token_count:
                stage_stats.tokens_total += token_count
                stage_stats.inference_seconds_total += latency_ms / 1000.0

            agent_stats = self._agents.setdefault(agent_name, _AgentStats())
            agent_stats.total += 1
            if status == "error":
                agent_stats.errors += 1

    def reset(self) -> None:
        """Vacía todo el estado acumulado -- para tests; en producción el
        agregador vive mientras viva el proceso."""
        with self._lock:
            self._stages.clear()
            self._agents.clear()

    def get_performance_summary(self) -> dict:
        """Snapshot inmutable del estado actual: percentiles de latencia por
        etapa, tokens/seg agregado (tokens totales de `model_inference`
        dividido por los segundos reales que tomó generarlos -- no un
        promedio de tasas por solicitud, que subestimaría solicitudes
        largas y lentas), y total/errores/tasa de fallo por agente.
        """
        with self._lock:
            por_stage = {}
            for stage, stats in self._stages.items():
                por_stage[stage] = {
                    "count": len(stats.latencies_ms),
                    "p50_ms": _percentile(stats.latencies_ms, 50),
                    "p95_ms": _percentile(stats.latencies_ms, 95),
                    "p99_ms": _percentile(stats.latencies_ms, 99),
                }

            tokens_total = sum(s.tokens_total for s in self._stages.values())
            inference_seconds_total = sum(s.inference_seconds_total for s in self._stages.values())
            tokens_per_second = (
                tokens_total / inference_seconds_total if inference_seconds_total > 0 else 0.0
            )

            por_agente = {}
            for agent_name, stats in self._agents.items():
                por_agente[agent_name] = {
                    "total": stats.total,
                    "errors": stats.errors,
                    "error_rate": (stats.errors / stats.total) if stats.total else 0.0,
                }

            return {
                "stages": por_stage,
                "tokens_total": tokens_total,
                "tokens_per_second": tokens_per_second,
                "agents": por_agente,
            }


# Instancia real del proceso: `trace_stage` (logger.py) escribe acá en cada
# span completado. Los tests que no quieren compartir estado global pueden
# instanciar su propio `MetricsAggregator()` en cambio.
default_aggregator = MetricsAggregator()
