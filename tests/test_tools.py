"""Tests de las herramientas industriales y del despacho de funciones del registro."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math

import pytest
from pydantic import ValidationError

from src.guardrails.validators import DangerousSQLError
from src.tools.industrial_tools import (
    CalculateRULInput,
    CalculateRULTool,
    QueryDuckDBInput,
    QueryDuckDBTool,
    SensorAnomalyCheckInput,
    SensorAnomalyCheckTool,
    build_default_registry,
)
from src.tools.registry import ToolExecutionError, ToolNotFoundError, ToolRegistry


# --------------------------------------------------------------------------- #
# QueryDuckDBTool
# --------------------------------------------------------------------------- #


def test_query_duckdb_tool_runs_select_and_returns_rows(tmp_path):
    database = str(tmp_path / "sensors.duckdb")
    populate_sensor_table(database)
    tool = QueryDuckDBTool(database=database)

    result = tool.run(QueryDuckDBInput(query="SELECT id, value FROM sensors ORDER BY id"))

    assert result["columns"] == ["id", "value"]
    assert result["rows"] == [[1, 10.5], [2, 20.0], [3, 30.25]]
    assert result["row_count"] == 3


def test_query_duckdb_tool_respects_limit(tmp_path):
    database = str(tmp_path / "sensors.duckdb")
    populate_sensor_table(database)
    tool = QueryDuckDBTool(database=database)

    result = tool.run(QueryDuckDBInput(query="SELECT id FROM sensors ORDER BY id", limit=2))

    assert result["row_count"] == 2
    assert result["rows"] == [[1], [2]]


def test_query_duckdb_tool_blocks_dangerous_query():
    tool = QueryDuckDBTool()

    with pytest.raises(DangerousSQLError):
        tool.run(QueryDuckDBInput(query="DROP TABLE sensors"))


def populate_sensor_table(database: str) -> None:
    import duckdb

    connection = duckdb.connect(database=database)
    try:
        connection.execute("CREATE TABLE sensors (id INTEGER, value DOUBLE)")
        connection.execute(
            "INSERT INTO sensors VALUES (1, 10.5), (2, 20.0), (3, 30.25)"
        )
    finally:
        connection.close()


# --------------------------------------------------------------------------- #
# SensorAnomalyCheckTool
# --------------------------------------------------------------------------- #


def test_sensor_anomaly_check_flags_outlier():
    tool = SensorAnomalyCheckTool()

    result = tool.run(SensorAnomalyCheckInput(readings=[10, 11, 9, 10, 10, 60], threshold=2.0))

    assert result["is_last_reading_anomalous"] is True
    assert result["anomaly_indices"] == [5]


def test_sensor_anomaly_check_no_anomaly_on_stable_readings():
    tool = SensorAnomalyCheckTool()

    result = tool.run(SensorAnomalyCheckInput(readings=[10, 10.1, 9.9, 10.05, 9.95], threshold=3.0))

    assert result["is_last_reading_anomalous"] is False
    assert result["anomaly_indices"] == []


def test_sensor_anomaly_check_handles_constant_series_without_division_by_zero():
    tool = SensorAnomalyCheckTool()

    result = tool.run(SensorAnomalyCheckInput(readings=[5, 5, 5, 5]))

    assert result["std_dev"] == 0.0
    assert all(z == 0.0 for z in result["z_scores"])
    assert result["is_last_reading_anomalous"] is False


def test_sensor_anomaly_check_input_requires_at_least_two_readings():
    with pytest.raises(ValidationError):
        SensorAnomalyCheckInput(readings=[1.0])


# --------------------------------------------------------------------------- #
# CalculateRULTool
# --------------------------------------------------------------------------- #


def test_calculate_rul_estimates_time_to_failure_for_linear_degradation():
    tool = CalculateRULTool()

    result = tool.run(
        CalculateRULInput(
            timestamps=[0, 1, 2, 3],
            measurements=[10, 20, 30, 40],
            failure_threshold=100,
        )
    )

    assert result["is_degrading_toward_failure"] is True
    assert math.isclose(result["rul_estimate"], 6.0, rel_tol=1e-6)


def test_calculate_rul_returns_none_when_not_degrading_toward_failure():
    tool = CalculateRULTool()

    result = tool.run(
        CalculateRULInput(
            timestamps=[0, 1, 2, 3],
            measurements=[40, 30, 20, 10],
            failure_threshold=100,
        )
    )

    assert result["rul_estimate"] is None
    assert result["is_degrading_toward_failure"] is False


def test_calculate_rul_input_rejects_mismatched_lengths():
    with pytest.raises(ValidationError):
        CalculateRULInput(timestamps=[0, 1, 2], measurements=[10, 20], failure_threshold=50)


# --------------------------------------------------------------------------- #
# ToolRegistry: despacho de funciones
# --------------------------------------------------------------------------- #


def test_registry_generates_openai_compatible_schema():
    registry = build_default_registry()

    schemas = registry.to_openai_schemas()
    names = {schema["function"]["name"] for schema in schemas}

    assert names == {"query_duckdb", "sensor_anomaly_check", "calculate_rul"}
    for schema in schemas:
        assert schema["type"] == "function"
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"


def test_registry_dispatch_with_dict_arguments():
    registry = build_default_registry()

    result = registry.dispatch(
        "sensor_anomaly_check", {"readings": [1, 1, 1, 1, 1, 20], "threshold": 2.0}
    )

    assert result["is_last_reading_anomalous"] is True


def test_registry_dispatch_with_json_string_arguments():
    registry = build_default_registry()

    payload = json.dumps({"readings": [1, 1, 1, 1, 1, 20], "threshold": 2.0})
    result = registry.dispatch("sensor_anomaly_check", payload)

    assert result["is_last_reading_anomalous"] is True


def test_registry_dispatch_unknown_tool_raises_not_found():
    registry = build_default_registry()

    with pytest.raises(ToolNotFoundError):
        registry.dispatch("unknown_tool", {})


def test_registry_dispatch_invalid_arguments_raise_execution_error():
    registry = build_default_registry()

    with pytest.raises(ToolExecutionError):
        registry.dispatch("calculate_rul", {"timestamps": [0, 1]})  # faltan campos requeridos


def test_registry_rejects_duplicate_registration():
    registry = ToolRegistry()
    registry.register_tool("t", "desc", SensorAnomalyCheckInput, lambda params: params)

    with pytest.raises(ValueError):
        registry.register_tool("t", "otra desc", SensorAnomalyCheckInput, lambda params: params)
