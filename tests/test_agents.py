"""Pruebas del pipeline multi-agente (src/agents): RouterAgent -> AnalyticsAgent -> VerifierAgent.

A diferencia de `tests/test_integration.py` (que ejercita el flujo completo
solo a través de la API HTTP), aquí se prueba cada agente por separado con
`LLMServer` mockeado, además del `AgentOrchestrator` de forma unitaria y, al
final, el cableado completo API -> `AgentOrchestrator` con `TestClient`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.agents import (
    AgentMessage,
    AgentOrchestrator,
    AnalyticsAgent,
    FaithfulnessError,
    RouterAgent,
    ToolCallEnvelope,
    VerifierAgent,
)
from src.engine import GenerationConfig, GenerationError
from src.guardrails import DangerousSQLError, OutputValidationError
from src.tools import ToolNotFoundError, build_default_registry

CONFIG = GenerationConfig()


# --------------------------------------------------------------------------- #
# RouterAgent
# --------------------------------------------------------------------------- #


def test_router_agent_returns_no_tool_call_for_plain_text():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "hola, ¿en qué puedo ayudarte?"
    router = RouterAgent(mock_llm, build_default_registry())

    decision = router.route([AgentMessage(role="user", content="hola")], config=CONFIG)

    assert decision.raw_text == "hola, ¿en qué puedo ayudarte?"
    assert decision.tool_call is None


def test_router_agent_parses_valid_tool_call():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = json.dumps(
        {"tool": "sensor_anomaly_check", "arguments": {"readings": [1, 2, 3]}}
    )
    router = RouterAgent(mock_llm, build_default_registry())

    decision = router.route([AgentMessage(role="user", content="revisa el sensor")], config=CONFIG)

    assert decision.tool_call is not None
    assert decision.tool_call.tool == "sensor_anomaly_check"
    assert decision.tool_call.arguments == {"readings": [1, 2, 3]}


def test_router_agent_prompt_lists_available_tools():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "ok"
    router = RouterAgent(mock_llm, build_default_registry())

    router.route([AgentMessage(role="user", content="hola")], config=CONFIG)

    prompt = mock_llm.generate.call_args.args[0]
    assert "calculate_rul" in prompt
    assert "query_duckdb" in prompt
    assert "sensor_anomaly_check" in prompt


# --------------------------------------------------------------------------- #
# AnalyticsAgent
# --------------------------------------------------------------------------- #


def test_analytics_agent_executes_real_tool_and_returns_tool_message():
    agent = AnalyticsAgent(build_default_registry())

    tool_call = ToolCallEnvelope(
        tool="sensor_anomaly_check", arguments={"readings": [10, 10.2, 9.8, 10.1, 60.0]}
    )
    result = agent.execute(tool_call)

    assert result.message.role == "tool"
    payload = json.loads(result.message.content)
    assert payload == result.raw_result
    assert "is_last_reading_anomalous" in payload
    assert "z_scores" in payload


def test_analytics_agent_propagates_tool_not_found():
    agent = AnalyticsAgent(build_default_registry())

    with pytest.raises(ToolNotFoundError):
        agent.execute(ToolCallEnvelope(tool="no_existe", arguments={}))


def test_analytics_agent_propagates_dangerous_sql_guardrail():
    agent = AnalyticsAgent(build_default_registry())

    with pytest.raises(DangerousSQLError):
        agent.execute(ToolCallEnvelope(tool="query_duckdb", arguments={"query": "DROP TABLE x"}))


# --------------------------------------------------------------------------- #
# VerifierAgent
# --------------------------------------------------------------------------- #


def test_verifier_agent_accepts_plain_answer_without_tool_context():
    result = VerifierAgent().verify("La presión nominal es de 4.5 bar.", tool_context=None)

    assert result.is_faithful is True
    assert result.is_valid_format is True


def test_verifier_agent_rejects_empty_response_as_generation_error():
    with pytest.raises(GenerationError):
        VerifierAgent().verify("   ", tool_context=None)


def test_verifier_agent_rejects_unresolved_tool_call_as_final_answer():
    unresolved = json.dumps({"tool": "query_duckdb", "arguments": {"query": "SELECT 1"}})

    with pytest.raises(OutputValidationError):
        VerifierAgent().verify(unresolved, tool_context=None)


def test_verifier_agent_accepts_response_grounded_in_tool_context():
    tool_context = {"rows": [[3, 95.0]], "row_count": 1}

    result = VerifierAgent().verify("El sensor 3 registra 95.0°C.", tool_context=tool_context)

    assert result.is_faithful is True


def test_verifier_agent_rejects_response_with_unsupported_numbers():
    tool_context = {"rows": [[1, 72.5]], "row_count": 1}

    with pytest.raises(FaithfulnessError):
        VerifierAgent().verify("El sensor reporta 999.0 grados.", tool_context=tool_context)


# --------------------------------------------------------------------------- #
# AgentOrchestrator (unitario: Router -> Analytics -> Verifier, sin pasar por la API)
# --------------------------------------------------------------------------- #


def test_orchestrator_skips_analytics_agent_when_no_tool_call():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "hola, ¿en qué puedo ayudarte?"
    orchestrator = AgentOrchestrator(mock_llm, build_default_registry())

    result = orchestrator.run([AgentMessage(role="user", content="hola")], config=CONFIG)

    assert result.text == "hola, ¿en qué puedo ayudarte?"
    assert result.used_tool is None
    mock_llm.generate.assert_called_once()


def test_orchestrator_runs_full_router_analytics_verifier_chain():
    tool_call = {"tool": "sensor_anomaly_check", "arguments": {"readings": [10, 10.1, 9.9, 40.0]}}
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        json.dumps(tool_call),
        "La última lectura es anómala respecto al histórico.",
    ]
    orchestrator = AgentOrchestrator(mock_llm, build_default_registry())

    result = orchestrator.run([AgentMessage(role="user", content="¿anomalía?")], config=CONFIG)

    assert result.used_tool == "sensor_anomaly_check"
    assert result.text == "La última lectura es anómala respecto al histórico."
    assert mock_llm.generate.call_count == 2

    second_prompt = mock_llm.generate.call_args_list[1].args[0]
    assert "is_last_reading_anomalous" in second_prompt


def test_orchestrator_raises_faithfulness_error_when_final_answer_invents_numbers():
    tool_call = {"tool": "sensor_anomaly_check", "arguments": {"readings": [10, 10.1, 9.9, 10.2]}}
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [
        json.dumps(tool_call),
        "La lectura llegó a 999.9 grados, muy por encima de lo normal.",
    ]
    orchestrator = AgentOrchestrator(mock_llm, build_default_registry())

    with pytest.raises(FaithfulnessError):
        orchestrator.run([AgentMessage(role="user", content="¿anomalía?")], config=CONFIG)


def test_orchestrator_rejects_unresolved_second_tool_call():
    first_call = {"tool": "sensor_anomaly_check", "arguments": {"readings": [1, 2, 3]}}
    second_call = {"tool": "calculate_rul", "arguments": {"timestamps": [0, 1]}}
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = [json.dumps(first_call), json.dumps(second_call)]
    orchestrator = AgentOrchestrator(mock_llm, build_default_registry())

    with pytest.raises(OutputValidationError):
        orchestrator.run([AgentMessage(role="user", content="hola")], config=CONFIG)


# --------------------------------------------------------------------------- #
# Flujo completo a través de la API (TestClient + LLMServer mockeado)
# --------------------------------------------------------------------------- #

from src.api.routes import app  # noqa: E402

client = TestClient(app)


def _payload(content: str) -> dict:
    return {"model": "local-slm", "messages": [{"role": "user", "content": content}]}


def test_api_delegates_plain_answer_to_agent_orchestrator():
    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.return_value = "La presión nominal es de 4.5 bar."
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=_payload("¿presión nominal?"))

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "La presión nominal es de 4.5 bar."


def test_api_runs_full_agent_chain_for_tool_call():
    tool_call = {"tool": "sensor_anomaly_check", "arguments": {"readings": [10, 10.1, 9.9, 40.0]}}

    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.side_effect = [
            json.dumps(tool_call),
            "La última lectura es anómala respecto al histórico.",
        ]
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=_payload("¿anomalía en el sensor?"))

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "La última lectura es anómala respecto al histórico."
    )
    assert mock_server.generate.call_count == 2


def test_api_rejects_final_answer_that_invents_numbers_not_in_tool_context():
    tool_call = {"tool": "sensor_anomaly_check", "arguments": {"readings": [10, 10.1, 9.9, 10.2]}}

    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.side_effect = [
            json.dumps(tool_call),
            "La lectura llegó a 999.9 grados.",
        ]
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=_payload("¿anomalía?"))

    assert response.status_code == 400


def test_api_rejects_unresolved_second_tool_call_as_final_answer():
    first_call = {"tool": "sensor_anomaly_check", "arguments": {"readings": [1, 2, 3]}}
    second_call = {"tool": "calculate_rul", "arguments": {"timestamps": [0, 1]}}

    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.side_effect = [json.dumps(first_call), json.dumps(second_call)]
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=_payload("hola"))

    assert response.status_code == 400
