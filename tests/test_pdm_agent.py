"""Tests del PdMAgent (`src/agents/pdm_agent.py`): interpretación de los
resultados crudos de `calculate_rul`/`sensor_anomaly_check` contra umbrales
de urgencia fijos, y comportamiento advisor (nunca lanza excepción, a
diferencia de `SafetyComplianceAgent`).
"""

from __future__ import annotations

import time

import pytest

from src.agents.pdm_agent import (
    DEFAULT_ANOMALY_URGENT_COUNT,
    DEFAULT_RUL_URGENT_THRESHOLD_HOURS,
    MaintenanceAssessment,
    PdMAgent,
)


@pytest.fixture()
def agent() -> PdMAgent:
    return PdMAgent()


# --------------------------------------------------------------------------- #
# calculate_rul
# --------------------------------------------------------------------------- #


def test_assess_flags_rul_at_or_below_threshold_as_urgent(agent):
    result = agent.assess("calculate_rul", {"rul_estimate": 10.0})

    assert isinstance(result, MaintenanceAssessment)
    assert result.is_urgent is True
    assert result.tool == "calculate_rul"
    assert result.rul_estimate == 10.0
    assert "10.0h" in result.reason


def test_assess_rul_exactly_at_threshold_is_urgent(agent):
    result = agent.assess("calculate_rul", {"rul_estimate": DEFAULT_RUL_URGENT_THRESHOLD_HOURS})

    assert result.is_urgent is True


def test_assess_rul_just_above_threshold_is_not_urgent(agent):
    result = agent.assess("calculate_rul", {"rul_estimate": DEFAULT_RUL_URGENT_THRESHOLD_HOURS + 0.1})

    assert result.is_urgent is False
    assert result.reason is None


def test_assess_rul_none_estimate_is_not_urgent():
    """rul_estimate=None significa que CalculateRULTool no proyecta falla
    (tendencia plana o alejándose del umbral) -- no es una alerta."""
    agent = PdMAgent()
    result = agent.assess("calculate_rul", {"rul_estimate": None, "is_degrading_toward_failure": False})

    assert result.is_urgent is False
    assert result.rul_estimate is None


# --------------------------------------------------------------------------- #
# sensor_anomaly_check
# --------------------------------------------------------------------------- #


def test_assess_flags_anomaly_count_at_or_above_threshold_as_urgent(agent):
    result = agent.assess("sensor_anomaly_check", {"anomaly_indices": [0, 3]})

    assert result.is_urgent is True
    assert result.tool == "sensor_anomaly_check"
    assert result.anomaly_count == 2
    assert str(DEFAULT_ANOMALY_URGENT_COUNT) in result.reason


def test_assess_anomaly_count_below_threshold_is_not_urgent(agent):
    result = agent.assess("sensor_anomaly_check", {"anomaly_indices": [0]})

    assert result.is_urgent is False
    assert result.anomaly_count == 1
    assert result.reason is None


def test_assess_anomaly_missing_key_defaults_to_zero_count(agent):
    result = agent.assess("sensor_anomaly_check", {})

    assert result.anomaly_count == 0
    assert result.is_urgent is False


# --------------------------------------------------------------------------- #
# Comportamiento advisor: nunca lanza, siempre devuelve un resultado
# --------------------------------------------------------------------------- #


def test_assess_returns_not_urgent_for_unrelated_tool(agent):
    result = agent.assess("query_duckdb", {"rows": [], "row_count": 0})

    assert result.is_urgent is False
    assert result.tool is None


def test_assess_returns_not_urgent_when_no_tool_was_used(agent):
    result = agent.assess(None, None)

    assert result.is_urgent is False
    assert result.tool is None
    assert result.reason is None


def test_assess_never_raises_regardless_of_input(agent):
    """A diferencia de SafetyComplianceAgent.audit (que lanza SafetyAlertError),
    PdMAgent es puramente advisor -- ninguna entrada debe producir una excepción."""
    for tool_name, payload in [
        ("calculate_rul", {}),
        ("sensor_anomaly_check", {}),
        ("calculate_rul", {"rul_estimate": -5.0}),
        ("unknown_tool", {"anything": True}),
    ]:
        result = agent.assess(tool_name, payload)
        assert isinstance(result, MaintenanceAssessment)


def test_assess_runs_without_added_latency(agent):
    """Chequeo local, sin red ni modelo -- debe resolver en microsegundos."""
    start = time.perf_counter()
    agent.assess("calculate_rul", {"rul_estimate": 5.0})
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05


# --------------------------------------------------------------------------- #
# Umbrales configurables por instancia (no por entorno ni por prompt)
# --------------------------------------------------------------------------- #


def test_custom_thresholds_are_respected():
    agent = PdMAgent(rul_urgent_threshold_hours=100.0, anomaly_urgent_count=1)

    assert agent.assess("calculate_rul", {"rul_estimate": 50.0}).is_urgent is True
    assert agent.assess("sensor_anomaly_check", {"anomaly_indices": [0]}).is_urgent is True
