"""Tests de guardrails: bloqueo de SQL peligroso y validación estricta de JSON de salida."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.guardrails.validators import (
    DangerousSQLError,
    OutputValidationError,
    validate_json_output,
    validate_sql_query,
)
from src.tools.industrial_tools import QueryDuckDBInput, SensorAnomalyCheckInput
from src.tools.registry import ToolExecutionError
from src.tools.industrial_tools import build_default_registry


# --------------------------------------------------------------------------- #
# validate_sql_query
# --------------------------------------------------------------------------- #


def test_allows_plain_select():
    assert validate_sql_query("SELECT * FROM sensors") == "SELECT * FROM sensors"


def test_allows_select_with_trailing_semicolon_and_whitespace():
    assert validate_sql_query("  SELECT id FROM sensors;  ") == "SELECT id FROM sensors"


def test_allows_cte_with_with_clause():
    query = "WITH recent AS (SELECT * FROM sensors) SELECT * FROM recent"
    assert validate_sql_query(query) == query


@pytest.mark.parametrize("keyword", ["DROP", "DELETE", "ALTER"])
def test_blocks_explicitly_required_dangerous_keywords(keyword):
    with pytest.raises(DangerousSQLError):
        validate_sql_query(f"{keyword} TABLE sensors")


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE sensors",
        "drop table sensors",
        "DeLeTe FROM sensors WHERE id = 1",
        "ALTER TABLE sensors ADD COLUMN x INT",
        "TRUNCATE TABLE sensors",
        "INSERT INTO sensors VALUES (1, 2)",
        "UPDATE sensors SET value = 0",
        "CREATE TABLE evil (x INT)",
        "ATTACH 'malicious.db' AS m",
        "EXEC sp_configure",
        "GRANT ALL ON sensors TO public",
    ],
)
def test_blocks_dangerous_and_mutating_statements_case_insensitive(query):
    with pytest.raises(DangerousSQLError):
        validate_sql_query(query)


def test_blocks_stacked_statements_sql_injection():
    with pytest.raises(DangerousSQLError):
        validate_sql_query("SELECT * FROM sensors; DROP TABLE sensors;")


def test_blocks_dangerous_keyword_hidden_inside_subquery():
    with pytest.raises(DangerousSQLError):
        validate_sql_query("SELECT * FROM (DELETE FROM sensors RETURNING *) AS t")


def test_blocks_dangerous_keyword_hidden_behind_line_comment():
    with pytest.raises(DangerousSQLError):
        validate_sql_query("SELECT * FROM sensors -- comentario\nDROP TABLE sensors")


def test_blocks_empty_query():
    with pytest.raises(DangerousSQLError):
        validate_sql_query("   ")


def test_blocks_non_string_query():
    with pytest.raises(DangerousSQLError):
        validate_sql_query(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# validate_json_output
# --------------------------------------------------------------------------- #


def test_validate_json_output_accepts_valid_payload():
    validated = validate_json_output(
        '{"readings": [1.0, 2.0, 3.0], "threshold": 2.5}', SensorAnomalyCheckInput
    )
    assert validated.readings == [1.0, 2.0, 3.0]
    assert validated.threshold == 2.5


def test_validate_json_output_rejects_malformed_json():
    with pytest.raises(OutputValidationError):
        validate_json_output('{"query": "SELECT 1"', QueryDuckDBInput)  # falta '}'


def test_validate_json_output_rejects_non_object_top_level():
    with pytest.raises(OutputValidationError):
        validate_json_output("[1, 2, 3]", QueryDuckDBInput)


def test_validate_json_output_rejects_schema_violation_missing_field():
    with pytest.raises(OutputValidationError):
        validate_json_output("{}", QueryDuckDBInput)  # falta 'query', requerido


def test_validate_json_output_rejects_unknown_extra_fields():
    with pytest.raises(OutputValidationError):
        validate_json_output(
            '{"query": "SELECT 1", "drop_everything": true}', QueryDuckDBInput
        )


def test_validate_json_output_rejects_wrong_type_in_strict_mode():
    with pytest.raises(OutputValidationError):
        # 'threshold' debe ser numérico; en modo estricto una cadena no se coacciona.
        validate_json_output(
            '{"readings": [1.0, 2.0, 3.0], "threshold": "high"}', SensorAnomalyCheckInput
        )


# --------------------------------------------------------------------------- #
# Integración: la capa de guardrails protege el despacho del registro
# --------------------------------------------------------------------------- #


def test_registry_dispatch_blocks_dangerous_sql_end_to_end():
    registry = build_default_registry()

    with pytest.raises(DangerousSQLError):
        registry.dispatch("query_duckdb", {"query": "DELETE FROM sensors"})


def test_registry_dispatch_blocks_malformed_json_arguments():
    registry = build_default_registry()

    with pytest.raises(ToolExecutionError):
        registry.dispatch("query_duckdb", "{not valid json")
