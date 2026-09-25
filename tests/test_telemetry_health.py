"""Pruebas del agregador de métricas de rendimiento (src/telemetry/metrics.py)
y del diagnóstico de salud del sistema (src/telemetry/health.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.telemetry import default_aggregator, trace_stage
from src.telemetry.health import (
    STATUS_DEGRADED,
    STATUS_HEALTHY,
    STATUS_UNHEALTHY,
    check_system_health,
)
from src.telemetry.metrics import MetricsAggregator


# ---------------------------------------------------------------------------
# MetricsAggregator: tokens/seg
# ---------------------------------------------------------------------------

def test_tokens_per_second_matches_the_formula_exactly():
    agg = MetricsAggregator()
    agg.record(stage="model_inference", agent_name="engine", latency_ms=500.0, status="ok", token_count=100)
    agg.record(stage="model_inference", agent_name="engine", latency_ms=500.0, status="ok", token_count=50)

    resumen = agg.get_performance_summary()

    # 150 tokens en 1.0s reales (500ms + 500ms) => 150 tok/s exacto, no un
    # promedio de dos tasas por solicitud (100/0.5 y 50/0.5 promediados
    # daria el mismo numero aca por simetria, pero con latencias distintas
    # promediar tasas subestimaria la solicitud lenta -- ver el test de abajo).
    assert resumen["tokens_total"] == 150
    assert resumen["tokens_per_second"] == pytest.approx(150.0)


def test_tokens_per_second_is_a_weighted_rate_not_an_average_of_rates():
    """Una solicitud rapida (10 tokens en 100ms = 100 tok/s) y una lenta
    (10 tokens en 900ms = 11.1 tok/s): promediar las dos tasas daria
    ~55.5 tok/s, pero la tasa real (20 tokens / 1.0s) es 20 tok/s."""
    agg = MetricsAggregator()
    agg.record(stage="model_inference", agent_name="engine", latency_ms=100.0, status="ok", token_count=10)
    agg.record(stage="model_inference", agent_name="engine", latency_ms=900.0, status="ok", token_count=10)

    resumen = agg.get_performance_summary()
    assert resumen["tokens_per_second"] == pytest.approx(20.0)


def test_tokens_per_second_is_zero_without_any_inference_yet():
    agg = MetricsAggregator()
    assert agg.get_performance_summary()["tokens_per_second"] == 0.0


def test_only_model_inference_stage_contributes_to_tokens_per_second():
    agg = MetricsAggregator()
    agg.record(stage="tokenizer", agent_name="engine", latency_ms=10.0, status="ok", token_count=999)
    resumen = agg.get_performance_summary()

    assert resumen["tokens_total"] == 0
    assert resumen["tokens_per_second"] == 0.0


# ---------------------------------------------------------------------------
# MetricsAggregator: percentiles
# ---------------------------------------------------------------------------

def test_percentiles_match_the_numpy_reference_exactly():
    agg = MetricsAggregator()
    latencias = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for lat in latencias:
        agg.record(stage="tokenizer", agent_name="engine", latency_ms=float(lat), status="ok")

    stage = agg.get_performance_summary()["stages"]["tokenizer"]
    assert stage["count"] == 10
    assert stage["p50_ms"] == pytest.approx(float(np.percentile(latencias, 50)))
    assert stage["p95_ms"] == pytest.approx(float(np.percentile(latencias, 95)))
    assert stage["p99_ms"] == pytest.approx(float(np.percentile(latencias, 99)))


def test_percentiles_are_broken_down_independently_per_stage():
    agg = MetricsAggregator()
    for _ in range(10):
        agg.record(stage="tokenizer", agent_name="engine", latency_ms=1.0, status="ok")
    for _ in range(10):
        agg.record(stage="model_inference", agent_name="engine", latency_ms=500.0, status="ok")

    stages = agg.get_performance_summary()["stages"]
    assert stages["tokenizer"]["p50_ms"] == pytest.approx(1.0)
    assert stages["model_inference"]["p50_ms"] == pytest.approx(500.0)


def test_percentile_is_none_without_any_observations():
    agg = MetricsAggregator()
    assert agg.get_performance_summary()["stages"] == {}


# ---------------------------------------------------------------------------
# MetricsAggregator: ventana acotada (memoria)
# ---------------------------------------------------------------------------

def test_latency_window_stays_bounded_and_does_not_grow_indefinitely():
    agg = MetricsAggregator(window_size=100)
    for i in range(5_000):
        agg.record(stage="tokenizer", agent_name="engine", latency_ms=float(i), status="ok")

    stage = agg.get_performance_summary()["stages"]["tokenizer"]
    assert stage["count"] == 100  # nunca crece mas alla de la ventana


def test_window_keeps_the_most_recent_observations_not_the_oldest():
    agg = MetricsAggregator(window_size=3)
    for lat in [1.0, 2.0, 3.0, 4.0, 5.0]:
        agg.record(stage="tokenizer", agent_name="engine", latency_ms=lat, status="ok")

    stage = agg.get_performance_summary()["stages"]["tokenizer"]
    assert stage["count"] == 3
    assert stage["p50_ms"] == pytest.approx(4.0)  # mediana de [3, 4, 5]


# ---------------------------------------------------------------------------
# MetricsAggregator: errores por agente
# ---------------------------------------------------------------------------

def test_error_rate_per_agent_is_computed_correctly():
    agg = MetricsAggregator()
    for _ in range(7):
        agg.record(stage="tool_execution", agent_name="analytics_agent", latency_ms=1.0, status="ok")
    for _ in range(3):
        agg.record(stage="tool_execution", agent_name="analytics_agent", latency_ms=1.0, status="error")

    stats = agg.get_performance_summary()["agents"]["analytics_agent"]
    assert stats["total"] == 10
    assert stats["errors"] == 3
    assert stats["error_rate"] == pytest.approx(0.3)


def test_trace_stage_automatically_feeds_the_default_aggregator():
    default_aggregator.reset()
    with trace_stage("model_inference", "engine") as span:
        span.token_count = 5

    resumen = default_aggregator.get_performance_summary()
    assert resumen["stages"]["model_inference"]["count"] == 1
    assert resumen["agents"]["engine"]["total"] == 1


# ---------------------------------------------------------------------------
# check_system_health: HEALTHY
# ---------------------------------------------------------------------------

def test_check_system_health_is_healthy_under_normal_conditions():
    agg = MetricsAggregator()
    agg.record(stage="model_inference", agent_name="engine", latency_ms=200.0, status="ok", token_count=50)
    agg.record(stage="post_processing", agent_name="verifier_agent", latency_ms=5.0, status="ok")

    llm = MagicMock()
    orchestrator = MagicMock()
    resultado = check_system_health(llm_server=llm, orchestrator=orchestrator, aggregator=agg)

    assert resultado["status"] == STATUS_HEALTHY
    assert "performance_summary" in resultado
    assert "checked_at" in resultado


def test_check_system_health_is_healthy_on_a_cold_start_with_no_data_yet():
    """Sin ninguna solicitud todavia no hay percentiles que evaluar -- eso
    no es degradacion, es falta de datos."""
    agg = MetricsAggregator()
    llm = MagicMock()
    resultado = check_system_health(llm_server=llm, orchestrator=MagicMock(), aggregator=agg)

    assert resultado["status"] == STATUS_HEALTHY


# ---------------------------------------------------------------------------
# check_system_health: DEGRADED
# ---------------------------------------------------------------------------

def test_check_system_health_is_degraded_when_p95_latency_exceeds_threshold():
    agg = MetricsAggregator()
    for _ in range(20):
        agg.record(stage="model_inference", agent_name="engine", latency_ms=3_000.0, status="ok", token_count=10)

    llm = MagicMock()
    resultado = check_system_health(
        llm_server=llm, orchestrator=MagicMock(), aggregator=agg, latency_p95_threshold_ms=2_000.0,
    )

    assert resultado["status"] == STATUS_DEGRADED
    latencia_component = next(c for c in resultado["components"] if c["name"] == "latency:model_inference")
    assert latencia_component["status"] == STATUS_DEGRADED


def test_a_non_critical_agent_failing_often_degrades_but_is_not_unhealthy():
    agg = MetricsAggregator()
    for _ in range(8):
        agg.record(stage="tool_execution", agent_name="analytics_agent", latency_ms=1.0, status="error")
    for _ in range(2):
        agg.record(stage="tool_execution", agent_name="analytics_agent", latency_ms=1.0, status="ok")

    llm = MagicMock()
    resultado = check_system_health(
        llm_server=llm, orchestrator=MagicMock(), aggregator=agg, error_rate_threshold=0.5,
    )

    assert resultado["status"] == STATUS_DEGRADED


# ---------------------------------------------------------------------------
# check_system_health: UNHEALTHY
# ---------------------------------------------------------------------------

def test_check_system_health_is_unhealthy_when_the_engine_does_not_respond():
    agg = MetricsAggregator()
    llm = MagicMock()
    llm.dry_run.side_effect = RuntimeError("contexto liberado")

    resultado = check_system_health(llm_server=llm, orchestrator=MagicMock(), aggregator=agg)

    assert resultado["status"] == STATUS_UNHEALTHY
    engine_component = next(c for c in resultado["components"] if c["name"] == "engine")
    assert engine_component["status"] == STATUS_UNHEALTHY
    assert "contexto liberado" in engine_component["detail"]


def test_check_system_health_is_unhealthy_without_a_loaded_model():
    agg = MetricsAggregator()
    resultado = check_system_health(llm_server=None, orchestrator=MagicMock(), aggregator=agg)

    assert resultado["status"] == STATUS_UNHEALTHY


def test_check_system_health_is_unhealthy_when_a_critical_agent_fails_often():
    """safety_agent y verifier_agent son las dos ultimas puertas bloqueantes
    del pipeline real (src/agents/orchestrator.py) -- una tasa de error alta
    en cualquiera de las dos tiene que bajar el veredicto a UNHEALTHY, no
    solo DEGRADED."""
    agg = MetricsAggregator()
    for _ in range(8):
        agg.record(stage="safety_audit", agent_name="safety_agent", latency_ms=1.0, status="error")
    for _ in range(2):
        agg.record(stage="safety_audit", agent_name="safety_agent", latency_ms=1.0, status="ok")

    llm = MagicMock()
    resultado = check_system_health(
        llm_server=llm, orchestrator=MagicMock(), aggregator=agg, error_rate_threshold=0.5,
    )

    assert resultado["status"] == STATUS_UNHEALTHY
    safety_component = next(c for c in resultado["components"] if c["name"] == "safety_agent")
    assert safety_component["status"] == STATUS_UNHEALTHY


def test_check_system_health_is_unhealthy_without_an_orchestrator():
    agg = MetricsAggregator()
    llm = MagicMock()
    resultado = check_system_health(llm_server=llm, orchestrator=None, aggregator=agg)

    assert resultado["status"] == STATUS_UNHEALTHY
    nombres_agentes = {"router", "analytics_agent", "maintenance_advisor", "verifier_agent",
                        "safety_agent", "pdm_agent"}
    componentes_unhealthy = {c["name"] for c in resultado["components"] if c["status"] == STATUS_UNHEALTHY}
    assert nombres_agentes <= componentes_unhealthy


# ---------------------------------------------------------------------------
# integración HTTP: /health/detailed
# ---------------------------------------------------------------------------

def test_health_detailed_endpoint_returns_a_real_diagnostic():
    from fastapi.testclient import TestClient

    from src.api.routes import app

    client = TestClient(app)
    response = client.get("/health/detailed")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {STATUS_HEALTHY, STATUS_DEGRADED, STATUS_UNHEALTHY}
    assert "components" in body and "performance_summary" in body


def test_plain_health_endpoint_is_unaffected_and_still_a_bare_liveness_probe():
    """/health/detailed es nuevo y aditivo -- /health sigue siendo el
    liveness probe simple que usa Docker, sin cambiar su contrato."""
    from fastapi.testclient import TestClient

    from src.api.routes import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
