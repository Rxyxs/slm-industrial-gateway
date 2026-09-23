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


# --------------------------------------------------------------------------- #
# /v1/pdm/diagnose -- no pasa por el SLM ni el orquestador de chat, así que
# estos tests no mockean nada: PdMAgent.diagnose() y su ToolRegistry
# subyacente (calculate_rul, en memoria) son pura computación local.
# --------------------------------------------------------------------------- #


def test_pdm_diagnose_flags_critical_asset() -> None:
    payload = {
        "asset_id": "SAG-01",
        "series": [
            {
                "metric": "vibration",
                "timestamps": [0, 1, 2, 3],
                "measurements": [10, 11, 12, 13],
                "failure_threshold": 23,
            }
        ],
    }

    response = client.post("/v1/pdm/diagnose", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["asset_id"] == "SAG-01"
    assert body["overall_status"] == "critical"
    assert body["bottleneck_metric"] == "vibration"
    assert body["recommended_maintenance_window_hours"] == pytest.approx(10.0, rel=1e-6)
    assert len(body["metric_diagnoses"]) == 1


def test_pdm_diagnose_healthy_asset() -> None:
    payload = {
        "asset_id": "PUMP-07",
        "series": [
            {
                "metric": "load",
                "timestamps": [0, 1, 2, 3],
                "measurements": [50.0, 50.01, 50.02, 50.03],
                "failure_threshold": 90.0,
            }
        ],
    }

    response = client.post("/v1/pdm/diagnose", json=payload)

    assert response.status_code == 200
    assert response.json()["overall_status"] == "healthy"


def test_pdm_diagnose_rejects_empty_series_list() -> None:
    response = client.post("/v1/pdm/diagnose", json={"asset_id": "SAG-02", "series": []})

    assert response.status_code == 400


def test_pdm_diagnose_multi_metric_picks_worst_as_bottleneck() -> None:
    payload = {
        "asset_id": "SAG-03",
        "series": [
            {"metric": "vibration", "timestamps": [0, 1, 2, 3], "measurements": [10, 11, 12, 13], "failure_threshold": 23},
            {"metric": "load", "timestamps": [0, 1, 2, 3], "measurements": [50.0, 50.01, 50.02, 50.03], "failure_threshold": 90.0},
        ],
    }

    response = client.post("/v1/pdm/diagnose", json=payload)

    body = response.json()
    assert body["overall_status"] == "critical"
    assert body["bottleneck_metric"] == "vibration"
    assert len(body["metric_diagnoses"]) == 2


# --------------------------------------------------------------------------- #
# Formato de error uniforme, request_id y red de seguridad para excepciones
# no capturadas -- ver `request_context_middleware`/`unhandled_exception_handler`.
# --------------------------------------------------------------------------- #


def test_error_response_has_uniform_error_and_type_shape() -> None:
    response = client.post("/v1/chat/completions", json={"model": "local-slm", "messages": []})

    assert response.status_code == 400
    body = response.json()
    assert set(body.keys()) == {"error", "type"}
    assert body["type"] == "ValueError"
    assert "detail" not in body  # no el shape por defecto de FastAPI


def test_model_load_error_response_has_uniform_shape_with_real_type() -> None:
    payload = {"model": "local-slm", "messages": [{"role": "user", "content": "hola"}]}

    with patch("src.api.routes.get_llm_server", side_effect=ModelLoadError("sin modelo")):
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 503
    body = response.json()
    assert body == {"error": "sin modelo", "type": "ModelLoadError"}


def test_pdm_diagnose_error_response_has_uniform_shape() -> None:
    response = client.post("/v1/pdm/diagnose", json={"asset_id": "SAG-99", "series": []})

    assert response.status_code == 400
    body = response.json()
    assert set(body.keys()) == {"error", "type"}
    assert body["type"] == "ValueError"


def test_response_carries_a_request_id_header() -> None:
    response = client.get("/health")

    assert "x-request-id" in {k.lower() for k in response.headers}


def test_request_id_from_client_is_echoed_back() -> None:
    response = client.get("/health", headers={"X-Request-ID": "client-supplied-id-123"})

    assert response.headers["x-request-id"] == "client-supplied-id-123"


def test_two_requests_without_a_client_id_get_different_request_ids() -> None:
    first = client.get("/health").headers["x-request-id"]
    second = client.get("/health").headers["x-request-id"]

    assert first != second


def test_unhandled_exception_returns_clean_json_500_not_a_traceback_page() -> None:
    """Un error que ningun `except` especifico del endpoint atrapa (acá,
    forzado dentro del propio orquestador) no debe tumbar la respuesta en la
    pagina de traceback por defecto de Starlette -- debe quedar en el mismo
    formato uniforme que cualquier otro error del gateway.

    `raise_server_exceptions=False`: el `ServerErrorMiddleware` de Starlette
    manda la respuesta del handler al cliente y DESPUES re-lanza la excepcion
    (para que el logging del servidor ASGI real -- uvicorn -- la vea); con el
    TestClient por defecto esa re-elevación llega tal cual al test, en vez de
    dejar inspeccionar la respuesta que un cliente real sí recibió."""
    local_client = TestClient(app, raise_server_exceptions=False)
    payload = {"model": "local-slm", "messages": [{"role": "user", "content": "hola"}]}

    with patch("src.api.routes.get_llm_server", side_effect=RuntimeError("fallo inesperado no mapeado")):
        response = local_client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 500
    body = response.json()
    assert body == {"error": "Error interno del servidor.", "type": "RuntimeError"}
