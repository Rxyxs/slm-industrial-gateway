"""Pruebas de integración end-to-end del gateway (API + engine + tools + guardrails).

A diferencia de `tests/test_api.py` (que sólo cubre el contrato HTTP con el
motor de inferencia simulado), estas pruebas ejercitan el flujo completo del
agente industrial descrito en `src/api/routes.py`:

    validación de entrada -> inferencia SLM -> ejecución de tool (si aplica)
    -> guardrails de salida -> respuesta JSON

El único componente simulado es el modelo (`LLMServer.generate`), ya que no
hay pesos GGUF disponibles en CI. El registro de herramientas, DuckDB y los
guardrails corren de verdad para verificar que las piezas de los distintos
módulos (`src.engine`, `src.tools`, `src.guardrails`) quedan bien conectadas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.api.routes import app

client = TestClient(app)


def _payload(user_content: str) -> dict:
    return {
        "model": "local-slm",
        "messages": [{"role": "user", "content": user_content}],
    }


# --------------------------------------------------------------------------- #
# Flujo sin tool: el SLM responde directamente en lenguaje natural
# --------------------------------------------------------------------------- #


def test_end_to_end_plain_answer_skips_tool_execution():
    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.return_value = "La presión nominal es de 4.5 bar."
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=_payload("¿Cuál es la presión nominal?"))

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "La presión nominal es de 4.5 bar."
    # Sin tool call, el SLM se invoca una única vez.
    mock_server.generate.assert_called_once()


# --------------------------------------------------------------------------- #
# Flujo con tool: query_duckdb real, guardrails reales, segunda pasada del SLM
# --------------------------------------------------------------------------- #


def test_end_to_end_dispatches_real_duckdb_tool_and_returns_final_answer():
    tool_call = {
        "tool": "query_duckdb",
        "arguments": {
            "query": (
                "SELECT * FROM (VALUES (1, 72.5), (2, 74.1), (3, 95.0)) "
                "AS t(sensor_id, temperature) ORDER BY sensor_id"
            )
        },
    }

    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.side_effect = [
            json.dumps(tool_call),
            "El sensor 3 registra 95.0°C, fuera de rango normal.",
        ]
        mock_get_server.return_value = mock_server

        response = client.post(
            "/v1/chat/completions",
            json=_payload("Revisa las temperaturas de los sensores 1, 2 y 3."),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == (
        "El sensor 3 registra 95.0°C, fuera de rango normal."
    )

    # El SLM se invoca dos veces: la propuesta de tool call y la respuesta final.
    assert mock_server.generate.call_count == 2

    # La segunda pasada debe incluir el resultado real de DuckDB (no simulado)
    # como contexto, confirmando que la tool corrió de verdad.
    second_prompt = mock_server.generate.call_args_list[1].args[0]
    assert "95.0" in second_prompt
    assert "row_count" in second_prompt


def test_end_to_end_dispatches_sensor_anomaly_tool():
    tool_call = {
        "tool": "sensor_anomaly_check",
        "arguments": {"readings": [10, 10.2, 9.8, 10.1, 60.0], "threshold": 2.0},
    }

    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.side_effect = [
            json.dumps(tool_call),
            "La última lectura es anómala respecto al histórico.",
        ]
        mock_get_server.return_value = mock_server

        response = client.post(
            "/v1/chat/completions", json=_payload("¿La última lectura del sensor es anómala?")
        )

    assert response.status_code == 200
    assert mock_server.generate.call_count == 2
    second_prompt = mock_server.generate.call_args_list[1].args[0]
    assert "is_last_reading_anomalous" in second_prompt


# --------------------------------------------------------------------------- #
# Guardrails de salida / seguridad protegiendo el despacho de tools
# --------------------------------------------------------------------------- #


def test_end_to_end_blocks_dangerous_sql_before_second_llm_call():
    tool_call = {"tool": "query_duckdb", "arguments": {"query": "DROP TABLE sensors"}}

    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.return_value = json.dumps(tool_call)
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=_payload("Borra la tabla de sensores."))

    assert response.status_code == 400
    # La consulta peligrosa se bloquea antes de pedirle al SLM una respuesta final.
    mock_server.generate.assert_called_once()


def test_end_to_end_unknown_tool_returns_400():
    tool_call = {"tool": "borrar_todo", "arguments": {}}

    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.return_value = json.dumps(tool_call)
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=_payload("hola"))

    assert response.status_code == 400


def test_end_to_end_invalid_tool_arguments_returns_400():
    # 'calculate_rul' requiere 'timestamps', 'measurements' y 'failure_threshold'.
    tool_call = {"tool": "calculate_rul", "arguments": {"timestamps": [0, 1]}}

    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.return_value = json.dumps(tool_call)
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=_payload("hola"))

    assert response.status_code == 400


def test_end_to_end_empty_final_answer_is_rejected_by_output_guardrail():
    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.return_value = "   "
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=_payload("hola"))

    assert response.status_code == 500


# --------------------------------------------------------------------------- #
# El registro de herramientas expuesto por la API es el registro real de src.tools
# --------------------------------------------------------------------------- #


def test_tool_registry_wired_into_api_matches_source_of_truth():
    from src.api.routes import get_tool_registry
    from src.tools import build_default_registry

    api_tool_names = get_tool_registry().list_tools()
    source_tool_names = build_default_registry().list_tools()

    assert api_tool_names == source_tool_names == ["calculate_rul", "query_duckdb", "sensor_anomaly_check"]
