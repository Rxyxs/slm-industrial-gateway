"""Tests del AnalyticsAgent: generación de argumentos para las 3 herramientas
industriales, y bloqueo de SQL destructivo antes de llegar a DuckDB."""

import pytest

from src.agents.analytics_agent import (
    ANALYTICS_REQUIRED,
    AnalyticsAgent,
    AnalyticsRequest,
    UnsupportedClassificationError,
    UnsupportedToolError,
)
from src.guardrails.validators import DangerousSQLError
from src.tools.industrial_tools import CalculateRULInput, QueryDuckDBInput, SensorAnomalyCheckInput


@pytest.fixture()
def agent() -> AnalyticsAgent:
    return AnalyticsAgent()


# --------------------------------------------------------------------------- #
# build_tool_call: generación de argumentos para las 3 herramientas
# --------------------------------------------------------------------------- #


def test_build_tool_call_query_duckdb(agent):
    request = AnalyticsRequest(
        classification=ANALYTICS_REQUIRED,
        tool_name="query_duckdb",
        parameters={"query": "SELECT 1 AS one", "limit": 50},
    )

    payload = agent.build_tool_call(request)

    assert payload.tool_name == "query_duckdb"
    assert payload.arguments["query"] == "SELECT 1 AS one"
    assert payload.arguments["limit"] == 50
    QueryDuckDBInput.model_validate(payload.arguments)  # el payload es válido para la herramienta real


def test_build_tool_call_sensor_anomaly_check(agent):
    request = AnalyticsRequest(
        classification=ANALYTICS_REQUIRED,
        tool_name="sensor_anomaly_check",
        parameters={"readings": [10, 10, 10, 10, 60], "threshold": 2.5},
    )

    payload = agent.build_tool_call(request)

    assert payload.tool_name == "sensor_anomaly_check"
    assert payload.arguments["readings"] == [10, 10, 10, 10, 60]
    assert payload.arguments["threshold"] == 2.5
    SensorAnomalyCheckInput.model_validate(payload.arguments)


def test_build_tool_call_calculate_rul(agent):
    request = AnalyticsRequest(
        classification=ANALYTICS_REQUIRED,
        tool_name="calculate_rul",
        parameters={
            "timestamps": [0, 1, 2, 3],
            "measurements": [10, 20, 30, 40],
            "failure_threshold": 100,
        },
    )

    payload = agent.build_tool_call(request)

    assert payload.tool_name == "calculate_rul"
    assert payload.arguments["failure_threshold"] == 100
    CalculateRULInput.model_validate(payload.arguments)


def test_build_tool_call_rejects_non_analytics_classification(agent):
    request = AnalyticsRequest(
        classification="OTHER_INTENT",
        tool_name="query_duckdb",
        parameters={"query": "SELECT 1"},
    )

    with pytest.raises(UnsupportedClassificationError):
        agent.build_tool_call(request)


def test_build_tool_call_rejects_unsupported_tool(agent):
    request = AnalyticsRequest(
        classification=ANALYTICS_REQUIRED,
        tool_name="not_a_real_tool",
        parameters={},
    )

    with pytest.raises(UnsupportedToolError):
        agent.build_tool_call(request)


def test_build_tool_call_rejects_invalid_parameters(agent):
    request = AnalyticsRequest(
        classification=ANALYTICS_REQUIRED,
        tool_name="sensor_anomaly_check",
        parameters={"readings": [1.0]},  # el schema real exige min_length=2
    )

    with pytest.raises(Exception):
        agent.build_tool_call(request)


# --------------------------------------------------------------------------- #
# handle_request: ejecución end-to-end + telemetría
# --------------------------------------------------------------------------- #


def test_handle_request_executes_query_duckdb_successfully(agent):
    request = AnalyticsRequest(
        classification=ANALYTICS_REQUIRED,
        tool_name="query_duckdb",
        parameters={"query": "SELECT 1 AS one"},
        request_id="req-1",
    )

    telemetry = agent.handle_request(request)

    assert telemetry.success is True
    assert telemetry.error is None
    assert telemetry.result["rows"] == [[1]]
    assert telemetry.request_id == "req-1"
    assert telemetry.latency_ms >= 0


def test_handle_request_executes_sensor_anomaly_check_successfully(agent):
    request = AnalyticsRequest(
        classification=ANALYTICS_REQUIRED,
        tool_name="sensor_anomaly_check",
        parameters={"readings": [10, 10, 10, 10, 60], "threshold": 2.0},
    )

    telemetry = agent.handle_request(request)

    assert telemetry.success is True
    assert telemetry.result["is_last_reading_anomalous"] is True


def test_handle_request_executes_calculate_rul_successfully(agent):
    request = AnalyticsRequest(
        classification=ANALYTICS_REQUIRED,
        tool_name="calculate_rul",
        parameters={"timestamps": [0, 1, 2, 3], "measurements": [10, 20, 30, 40], "failure_threshold": 100},
    )

    telemetry = agent.handle_request(request)

    assert telemetry.success is True
    assert telemetry.result["rul_estimate"] == pytest.approx(6.0)


@pytest.mark.parametrize("dangerous_query", ["DROP TABLE sensors", "DELETE FROM sensors WHERE id = 1"])
def test_handle_request_blocks_destructive_sql_and_returns_safe_error(agent, dangerous_query):
    request = AnalyticsRequest(
        classification=ANALYTICS_REQUIRED,
        tool_name="query_duckdb",
        parameters={"query": dangerous_query},
    )

    telemetry = agent.handle_request(request)

    assert telemetry.success is False
    assert telemetry.result is None
    assert telemetry.error is not None
    assert telemetry.error["type"] == DangerousSQLError.__name__


def test_handle_request_blocks_sql_before_reaching_the_registry(agent, monkeypatch):
    """No alcanza con que el resultado final sea un error: la interrupción
    tiene que pasar ANTES de despachar hacia el registro/DuckDB."""
    calls = []
    original_dispatch = agent.registry.dispatch

    def spy_dispatch(name, arguments):
        calls.append((name, arguments))
        return original_dispatch(name, arguments)

    monkeypatch.setattr(agent.registry, "dispatch", spy_dispatch)

    request = AnalyticsRequest(
        classification=ANALYTICS_REQUIRED,
        tool_name="query_duckdb",
        parameters={"query": "DROP TABLE sensors"},
    )

    telemetry = agent.handle_request(request)

    assert telemetry.success is False
    assert calls == []  # el registro nunca llegó a despachar


def test_handle_request_never_raises_on_unsupported_tool(agent):
    request = AnalyticsRequest(classification=ANALYTICS_REQUIRED, tool_name="drop_everything", parameters={})

    telemetry = agent.handle_request(request)

    assert telemetry.success is False
    assert telemetry.error["type"] == UnsupportedToolError.__name__


def test_handle_request_never_raises_on_wrong_classification(agent):
    request = AnalyticsRequest(classification="OTHER_INTENT", tool_name="query_duckdb", parameters={"query": "SELECT 1"})

    telemetry = agent.handle_request(request)

    assert telemetry.success is False
    assert telemetry.error["type"] == UnsupportedClassificationError.__name__
