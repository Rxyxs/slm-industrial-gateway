"""Tests del SafetyComplianceAgent (`src/agents/safety_agent.py`): rechazo de
recomendaciones que superan los límites de diseño del sitio, aprobación sin
latencia adicional de escenarios normales, e inmutabilidad de la matriz de
límites (regla fija, no configurable por entorno ni por prompt).
"""

from __future__ import annotations

import time

import pytest

from src.agents.safety_agent import (
    OPERATIONAL_LIMITS,
    OperationalLimit,
    SafetyAlertError,
    SafetyCheckResult,
    SafetyComplianceAgent,
)


@pytest.fixture()
def agent() -> SafetyComplianceAgent:
    return SafetyComplianceAgent()


# --------------------------------------------------------------------------- #
# Recomendaciones que superan un límite -- deben rechazarse
# --------------------------------------------------------------------------- #


def test_audit_rejects_power_above_design_limit(agent):
    with pytest.raises(SafetyAlertError) as excinfo:
        agent.audit("Se recomienda operar el generador a 180 MW para cubrir la demanda pico.")

    assert excinfo.value.code == "SAFETY_ALERT"
    assert len(excinfo.value.violations) == 1
    assert excinfo.value.violations[0].quantity == "potencia"
    assert excinfo.value.violations[0].value == 180.0


def test_audit_rejects_pressure_above_design_limit(agent):
    with pytest.raises(SafetyAlertError) as excinfo:
        agent.audit("Aumentar la presión del circuito hidráulico a 3500 PSI para forzar el desatoro.")

    assert excinfo.value.violations[0].quantity == "presión"
    assert excinfo.value.violations[0].value == 3500.0


def test_audit_rejects_temperature_above_critical_limit(agent):
    with pytest.raises(SafetyAlertError) as excinfo:
        agent.audit("Es seguro dejar que el rodamiento alcance 700°C durante el arranque.")

    assert excinfo.value.violations[0].quantity == "temperatura"
    assert excinfo.value.violations[0].value == 700.0


@pytest.mark.parametrize("phrase", ["720 grados C", "720 grados Celsius", "720°C", "720 ° C"])
def test_audit_recognizes_temperature_unit_variants(agent, phrase):
    with pytest.raises(SafetyAlertError):
        agent.audit(f"La temperatura sugerida es de {phrase}.")


def test_audit_reports_all_violations_when_multiple_limits_are_exceeded(agent):
    with pytest.raises(SafetyAlertError) as excinfo:
        agent.audit("Operar a 200 MW, 4000 PSI y 900°C simultáneamente.")

    quantities = {v.quantity for v in excinfo.value.violations}
    assert quantities == {"potencia", "presión", "temperatura"}


def test_audit_blocks_before_returning_any_partial_result(agent):
    """No alcanza con que se lance la excepción: no debe existir un
    `SafetyCheckResult` parcial marcado como seguro."""
    try:
        agent.audit("Operar el generador a 999 MW.")
        pytest.fail("Se esperaba SafetyAlertError")
    except SafetyAlertError as exc:
        assert "SAFETY_ALERT" in str(exc)
        assert exc.violations[0].limit == OPERATIONAL_LIMITS["power_mw"].max_value


# --------------------------------------------------------------------------- #
# Escenarios normales dentro de rango -- deben aprobarse, sin latencia extra
# --------------------------------------------------------------------------- #


def test_audit_approves_values_within_range(agent):
    result = agent.audit(
        "El generador opera a 90 MW, la presión del circuito es de 1800 PSI "
        "y la temperatura del rodamiento es de 65°C. Todo dentro de rango normal."
    )

    assert isinstance(result, SafetyCheckResult)
    assert result.is_safe is True
    assert ("potencia", 90.0, "MW") in result.checked_values
    assert ("presión", 1800.0, "PSI") in result.checked_values
    assert ("temperatura", 65.0, "°C") in result.checked_values


def test_audit_approves_response_with_no_numeric_values(agent):
    result = agent.audit("El equipo funciona correctamente, sin alertas activas.")

    assert result.is_safe is True
    assert result.checked_values == []


def test_audit_boundary_value_exactly_at_limit_is_approved(agent):
    """El límite de diseño es el máximo tolerado, no uno prohibido: exactamente
    en el límite todavía es una recomendación válida."""
    limit = OPERATIONAL_LIMITS["power_mw"].max_value

    result = agent.audit(f"Potencia máxima admisible: {limit:g} MW.")

    assert result.is_safe is True


def test_audit_just_above_boundary_is_rejected(agent):
    limit = OPERATIONAL_LIMITS["power_mw"].max_value

    with pytest.raises(SafetyAlertError):
        agent.audit(f"Potencia recomendada: {limit + 0.1:g} MW.")


def test_audit_runs_without_added_latency(agent):
    """Chequeo local (regex + comparación numérica), sin llamadas a red ni al
    modelo -- debe resolver en microsegundos, no en el orden de una
    inferencia real."""
    text = "El equipo opera a 90 MW, 1800 PSI y 65°C, todo dentro de rango." * 20

    start = time.perf_counter()
    result = agent.audit(text)
    elapsed = time.perf_counter() - start

    assert result.is_safe is True
    assert elapsed < 0.05


def test_audit_rejects_non_string_input(agent):
    with pytest.raises(TypeError):
        agent.audit(12345)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# La matriz de límites es fija e inmutable -- no configurable por entorno ni
# manipulable en runtime.
# --------------------------------------------------------------------------- #


def test_operational_limits_mapping_is_immutable():
    with pytest.raises(TypeError):
        OPERATIONAL_LIMITS["power_mw"] = OperationalLimit("otra", "MW", 1.0, "no debería poder asignarse")


def test_operational_limit_entries_are_frozen():
    limit = OPERATIONAL_LIMITS["pressure_psi"]
    with pytest.raises(Exception):
        limit.max_value = 999999.0  # type: ignore[misc]


def test_operational_limits_cover_power_pressure_and_temperature():
    assert set(OPERATIONAL_LIMITS) == {"power_mw", "pressure_psi", "temperature_c"}
    for limit in OPERATIONAL_LIMITS.values():
        assert limit.max_value > 0
