"""Pruebas del logging estructurado en JSON y los spans de latencia por
etapa (src/telemetry)."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.orchestrator import AgentOrchestrator
from src.agents.schemas import AgentMessage
from src.engine.llm_server import GenerationConfig, LLMServer
from src.telemetry import current_trace_id, start_trace, trace_stage
from src.telemetry.logger import logger as telemetry_logger

REQUIRED_FIELDS = {
    "timestamp", "trace_id", "span_id", "agent_name", "stage",
    "latency_ms", "token_count", "status", "message",
}


class _JSONCapture(logging.Handler):
    """Handler que aplica el mismo formatter real del logger de telemetría y
    guarda cada línea ya parseada -- prueba el JSON que de verdad sale por
    el handler configurado, no una reconstrucción a mano."""

    def __init__(self, formatter: logging.Formatter):
        super().__init__()
        self.setFormatter(formatter)
        self.records: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(json.loads(self.format(record)))


@pytest.fixture
def captura():
    formatter = telemetry_logger.handlers[0].formatter
    handler = _JSONCapture(formatter)
    telemetry_logger.addHandler(handler)
    try:
        yield handler
    finally:
        telemetry_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# esquema JSON
# ---------------------------------------------------------------------------

def test_trace_stage_emits_valid_json_with_all_required_fields(captura):
    with trace_stage("model_inference", "engine") as span:
        span.token_count = 42

    assert len(captura.records) == 1
    evento = captura.records[0]
    assert REQUIRED_FIELDS <= evento.keys()
    assert evento["stage"] == "model_inference"
    assert evento["agent_name"] == "engine"
    assert evento["status"] == "ok"
    assert evento["token_count"] == 42
    assert isinstance(evento["latency_ms"], (int, float))


def test_timestamp_is_iso_8601_utc(captura):
    with trace_stage("post_processing", "verifier_agent"):
        pass

    timestamp = captura.records[0]["timestamp"]
    # 'Z' o '+00:00' -- termina siendo parseable y en UTC de cualquier forma.
    from datetime import datetime
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_token_count_is_omitted_when_never_set(captura):
    with trace_stage("guardrail_input", "router"):
        pass

    assert "token_count" not in captura.records[0] or captura.records[0]["token_count"] is None


# ---------------------------------------------------------------------------
# atribución de latencia y trace_id/span_id
# ---------------------------------------------------------------------------

def test_latency_reflects_the_actual_time_spent_in_the_block(captura):
    with trace_stage("model_inference", "engine"):
        time.sleep(0.05)

    assert captura.records[0]["latency_ms"] >= 45  # 50ms nominal, margen por jitter del SO


def test_each_stage_gets_its_own_span_id_under_the_same_trace_id(captura):
    start_trace()
    with trace_stage("tokenizer", "engine"):
        pass
    with trace_stage("model_inference", "engine"):
        pass

    assert len(captura.records) == 2
    trace_ids = {r["trace_id"] for r in captura.records}
    span_ids = {r["span_id"] for r in captura.records}
    assert len(trace_ids) == 1  # mismo trace_id: son parte de la misma solicitud
    assert len(span_ids) == 2  # pero cada etapa tiene su propio span_id


def test_start_trace_can_be_given_an_explicit_id():
    tid = start_trace("un-trace-id-fijo")
    assert tid == "un-trace-id-fijo"
    assert current_trace_id() == "un-trace-id-fijo"


def test_trace_stage_used_as_a_decorator_still_measures_latency_and_status(captura):
    @trace_stage("post_processing", "verifier_agent")
    def procesar():
        time.sleep(0.02)

    procesar()

    assert len(captura.records) == 1
    assert captura.records[0]["status"] == "ok"
    assert captura.records[0]["latency_ms"] >= 15


# ---------------------------------------------------------------------------
# fallos: status=error, detalle de la excepción, trace_id no se rompe
# ---------------------------------------------------------------------------

def test_an_exception_inside_the_block_is_logged_as_error_and_reraised(captura):
    class FalloDeAgente(RuntimeError):
        pass

    with pytest.raises(FalloDeAgente, match="algo salio mal"):
        with trace_stage("tool_execution", "analytics_agent"):
            raise FalloDeAgente("algo salio mal")

    evento = captura.records[0]
    assert evento["status"] == "error"
    assert "exception" in evento
    assert "FalloDeAgente" in evento["exception"]
    assert "algo salio mal" in evento["exception"]


def test_trace_id_survives_a_failed_stage_and_keeps_correlating_later_stages(captura):
    start_trace("trace-que-no-se-rompe")

    with pytest.raises(ValueError):
        with trace_stage("tool_execution", "analytics_agent"):
            raise ValueError("fallo intencional")

    with trace_stage("safety_audit", "safety_agent"):
        pass

    assert captura.records[0]["status"] == "error"
    assert captura.records[0]["trace_id"] == "trace-que-no-se-rompe"
    assert captura.records[1]["status"] == "ok"
    assert captura.records[1]["trace_id"] == "trace-que-no-se-rompe"


def test_current_trace_id_auto_generates_one_when_nothing_started_it():
    """Un test aislado (sin start_trace explícito) no debe explotar ni
    devolver None -- current_trace_id() siempre entrega algo usable."""
    import src.telemetry.logger as telemetry_module
    telemetry_module._trace_id_var.set(None)

    tid = current_trace_id()
    assert isinstance(tid, str) and len(tid) > 0


# ---------------------------------------------------------------------------
# instrumentación real: LLMServer.generate()
# ---------------------------------------------------------------------------

@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.Q4_K_M.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 16)
    return path


def _build_server(model_file: Path, llm_instance: MagicMock) -> LLMServer:
    mock_llama_cls = MagicMock(return_value=llm_instance)
    with patch("src.engine.llm_server.Llama", mock_llama_cls):
        return LLMServer(model_file, n_gpu_layers=0)


def test_generate_emits_a_tokenizer_and_a_model_inference_span(model_file, captura):
    llm_instance = MagicMock()
    llm_instance.tokenize.return_value = [1, 2, 3, 4, 5]
    llm_instance.return_value = {
        "choices": [{"text": "hola", "index": 0, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 7},
    }
    server = _build_server(model_file, llm_instance)

    server.generate("hola mundo")

    stages = {r["stage"]: r for r in captura.records}
    assert "tokenizer" in stages and "model_inference" in stages
    assert stages["tokenizer"]["token_count"] == 5
    assert stages["model_inference"]["token_count"] == 7
    assert stages["tokenizer"]["agent_name"] == "engine"
    # Las dos etapas de una misma llamada a generate() comparten trace_id.
    assert stages["tokenizer"]["trace_id"] == stages["model_inference"]["trace_id"]


def test_generate_backend_failure_marks_the_model_inference_span_as_error(model_file, captura):
    llm_instance = MagicMock()
    llm_instance.tokenize.return_value = [1, 2]
    llm_instance.side_effect = RuntimeError("contexto agotado")
    server = _build_server(model_file, llm_instance)

    from src.engine.llm_server import GenerationError
    with pytest.raises(GenerationError):
        server.generate("hola")

    stages = {r["stage"]: r for r in captura.records}
    assert stages["tokenizer"]["status"] == "ok"
    assert stages["model_inference"]["status"] == "error"


# ---------------------------------------------------------------------------
# instrumentación real: AgentOrchestrator.run()
# ---------------------------------------------------------------------------

def test_orchestrator_run_emits_a_post_processing_span_for_the_verifier(captura):
    llm_server = MagicMock()
    llm_server.generate.return_value = "Respuesta final sin tool call."
    tool_registry = MagicMock()
    tool_registry.list_tools.return_value = []

    orchestrator = AgentOrchestrator(llm_server, tool_registry)
    orchestrator.run([AgentMessage(role="user", content="hola")], GenerationConfig())

    stages = [r["stage"] for r in captura.records]
    assert "post_processing" in stages
    assert "guardrail_input" in stages
    assert "safety_audit" in stages


def test_all_spans_within_one_orchestrator_run_share_the_same_trace_id(captura):
    llm_server = MagicMock()
    llm_server.generate.return_value = "Respuesta final sin tool call."
    tool_registry = MagicMock()
    tool_registry.list_tools.return_value = []

    orchestrator = AgentOrchestrator(llm_server, tool_registry)
    orchestrator.run([AgentMessage(role="user", content="hola")], GenerationConfig())

    trace_ids = {r["trace_id"] for r in captura.records}
    assert len(trace_ids) == 1
    span_ids = [r["span_id"] for r in captura.records]
    assert len(span_ids) == len(set(span_ids))  # cada span, uno distinto
