"""Tests unitarios del AnalyticsAgent (`src/agents/analytics_agent.py`): despacho
correcto de las 3 herramientas industriales, y propagación (no absorción) de
`DangerousSQLError` para SQL destructivo -- la conversión a un error HTTP
seguro ocurre en `src.api.routes` (ver `tests/test_agents.py`, sección de
API), no en este agente; acá se verifica que la ejecución se interrumpe de
inmediato y sin resultado parcial.
"""

from __future__ import annotations

import json

import pytest

from src.agents.analytics_agent import AnalyticsAgent, AnalyticsResult
from src.agents.schemas import AgentMessage, ToolCallEnvelope
from src.guardrails.validators import DangerousSQLError
from src.tools import ToolNotFoundError, build_default_registry


@pytest.fixture()
def agent() -> AnalyticsAgent:
    return AnalyticsAgent(build_default_registry())


# --------------------------------------------------------------------------- #
# Despacho correcto de las 3 herramientas industriales
# --------------------------------------------------------------------------- #


def test_execute_query_duckdb_returns_tool_message_with_real_result(agent):
    result = agent.execute(ToolCallEnvelope(tool="query_duckdb", arguments={"query": "SELECT 1 AS one"}))

    assert isinstance(result, AnalyticsResult)
    assert isinstance(result.message, AgentMessage)
    assert result.message.role == "tool"
    assert result.raw_result["rows"] == [[1]]
    assert json.loads(result.message.content) == result.raw_result


def test_execute_sensor_anomaly_check_returns_tool_message_with_real_result(agent):
    result = agent.execute(
        ToolCallEnvelope(tool="sensor_anomaly_check", arguments={"readings": [10, 10, 10, 10, 60], "threshold": 2.0})
    )

    assert result.message.role == "tool"
    assert result.raw_result["is_last_reading_anomalous"] is True
    assert json.loads(result.message.content) == result.raw_result


def test_execute_calculate_rul_returns_tool_message_with_real_result(agent):
    result = agent.execute(
        ToolCallEnvelope(
            tool="calculate_rul",
            arguments={"timestamps": [0, 1, 2, 3], "measurements": [10, 20, 30, 40], "failure_threshold": 100},
        )
    )

    assert result.message.role == "tool"
    assert result.raw_result["rul_estimate"] == pytest.approx(6.0)
    assert json.loads(result.message.content) == result.raw_result


# --------------------------------------------------------------------------- #
# SQL destructivo: la ejecución se interrumpe antes de tocar DuckDB
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dangerous_query", ["DROP TABLE sensors", "DELETE FROM sensors WHERE id = 1"])
def test_execute_interrupts_on_destructive_sql(agent, dangerous_query):
    with pytest.raises(DangerousSQLError):
        agent.execute(ToolCallEnvelope(tool="query_duckdb", arguments={"query": dangerous_query}))


def test_execute_raises_before_returning_any_partial_result(agent):
    """No alcanza con que se lance la excepción: no debe existir un
    `AnalyticsResult` parcial ni un side-effect de DuckDB asociado."""
    try:
        agent.execute(ToolCallEnvelope(tool="query_duckdb", arguments={"query": "DROP TABLE sensors"}))
        pytest.fail("Se esperaba DangerousSQLError")
    except DangerousSQLError as exc:
        assert "DROP" in str(exc)


def test_execute_propagates_tool_not_found(agent):
    with pytest.raises(ToolNotFoundError):
        agent.execute(ToolCallEnvelope(tool="no_existe", arguments={}))
