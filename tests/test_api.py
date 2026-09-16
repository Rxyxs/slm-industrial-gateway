"""Pruebas de integración de la API HTTP compatible con OpenAI (src/api)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.api.routes import app
from src.engine import GenerationError, ModelLoadError

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models() -> None:
    response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["object"] == "model"
    assert body["data"][0]["id"]


def test_metrics_endpoint_exposed() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_chat_completions_rejects_empty_messages() -> None:
    response = client.post(
        "/v1/chat/completions", json={"model": "local-slm", "messages": []}
    )

    assert response.status_code == 400


def test_chat_completions_contract() -> None:
    payload = {
        "model": "local-slm",
        "messages": [{"role": "user", "content": "hola mundo"}],
    }

    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.return_value = "hola, ¿en qué puedo ayudarte?"
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "local-slm"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "hola, ¿en qué puedo ayudarte?"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    mock_server.generate.assert_called_once()


def test_chat_completions_maps_model_load_error_to_503() -> None:
    payload = {
        "model": "local-slm",
        "messages": [{"role": "user", "content": "hola"}],
    }

    with patch("src.api.routes.get_llm_server", side_effect=ModelLoadError("sin modelo")):
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 503


def test_chat_completions_maps_generation_error_to_500() -> None:
    payload = {
        "model": "local-slm",
        "messages": [{"role": "user", "content": "hola"}],
    }

    with patch("src.api.routes.get_llm_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.generate.side_effect = GenerationError("fallo de inferencia")
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 500


def test_chat_completions_records_prometheus_metrics() -> None:
    payload = {
        "model": "local-slm",
        "messages": [{"role": "user", "content": "hola"}],
    }

    with patch("src.api.routes.get_llm_server") as mock_get_server, patch(
        "src.api.routes.TOKENS_GENERATED_TOTAL"
    ) as tokens_metric, patch(
        "src.api.routes.TOKEN_LATENCY_SECONDS"
    ) as latency_metric, patch(
        "src.api.routes.COMPLETION_REQUESTS_TOTAL"
    ) as requests_metric:
        mock_server = MagicMock()
        mock_server.generate.return_value = "respuesta generada"
        mock_get_server.return_value = mock_server

        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    tokens_metric.labels.assert_called_once_with(model="local-slm")
    tokens_metric.labels.return_value.inc.assert_called_once()
    latency_metric.labels.assert_called_once_with(model="local-slm")
    latency_metric.labels.return_value.observe.assert_called_once()
    requests_metric.labels.assert_called_once_with(model="local-slm", status="success")
    requests_metric.labels.return_value.inc.assert_called_once()


def test_chat_completions_records_error_metric_on_failure() -> None:
    payload = {
        "model": "local-slm",
        "messages": [{"role": "user", "content": "hola"}],
    }

    with patch(
        "src.api.routes.get_llm_server", side_effect=ModelLoadError("sin modelo")
    ), patch("src.api.routes.COMPLETION_REQUESTS_TOTAL") as requests_metric:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 503
    requests_metric.labels.assert_called_once_with(model="local-slm", status="error")
    requests_metric.labels.return_value.inc.assert_called_once()
