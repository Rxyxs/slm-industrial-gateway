"""Pruebas de utilidad contra un SLM real servido por Ollama.

A diferencia de tests/gateway (determinista, sin red), estas pruebas preguntan si el modelo
*sirve*: con el contexto inyectado, ¿propone subir la amina cuando la regla operativa lo
pide, y se abstiene cuando la planta está tranquila? Se saltan si no hay Ollama.

    OLLAMA_URL=http://127.0.0.1:11434 UTILITY_MODEL=qwen2.5:1.5b-instruct pytest tests/evals -v

Variables:
  UTILITY_MODEL           modelo a evaluar (default qwen2.5:1.5b-instruct)
  UTILITY_LIMIT           casos a usar, en tercios ADJUST / HOLD / near-miss (default: los 120)
  UTILITY_DEADLINE_S      deadline por decisión (default 30: mide calidad, no latencia;
                          en CPU sin GPU una decisión tarda varios segundos)
  UTILITY_MODE            prompt_mode del router (default context_gated, el recomendado)
  UTILITY_MIN_RECALL      % mínimo de horas ADJUST con el ajuste correcto ejecutable (default 80)
  UTILITY_MAX_FALSE_ADJ   % máximo de horas tranquilas o near-miss en que se ajusta algo (default 5)

Con qwen2.5:1.5b en GPU y context_gated se midió 100% / 0% / 0% sobre los 120 casos
(outputs/reports/utility_eval.json); los umbrales dejan margen para la variación de CPU.
"""

from __future__ import annotations

import os

import httpx
import pytest

from src.gateway.audit import AuditLog
from src.gateway.backends import OllamaBackend
from src.gateway.config import BackendSettings, GatewaySettings
from src.gateway.router import InferenceRouter
from src.gateway.utility_eval import load_cases, outcome_from, summarize

URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("UTILITY_MODEL", "qwen2.5:1.5b-instruct")
LIMIT = int(os.environ["UTILITY_LIMIT"]) if os.environ.get("UTILITY_LIMIT") else None
DEADLINE = float(os.environ.get("UTILITY_DEADLINE_S", "30"))
MODE = os.environ.get("UTILITY_MODE", "context_gated")
MIN_RECALL = float(os.environ.get("UTILITY_MIN_RECALL", "80"))
MAX_FALSE_ADJ = float(os.environ.get("UTILITY_MAX_FALSE_ADJ", "5"))

pytestmark = pytest.mark.slm


def _ollama_has_model() -> bool:
    try:
        tags = httpx.get(f"{URL}/api/tags", timeout=3).json()
    except (httpx.HTTPError, ValueError):
        return False
    return any(m["name"] == MODEL for m in tags.get("models", []))


@pytest.fixture(scope="module")
def summary():
    if not _ollama_has_model():
        pytest.skip(f"Ollama at {URL} does not serve {MODEL}")
    import asyncio

    async def run():
        backend = OllamaBackend(BackendSettings(name=MODEL, kind="ollama", base_url=URL, model=MODEL,
                                                tier="edge", timeout_s=DEADLINE))
        await backend.warmup()
        settings = GatewaySettings(deadline_s=DEADLINE, breaker_failure_threshold=10_000,
                                   audit_db_path=":memory:", prompt_mode=MODE)
        router = InferenceRouter(settings, [backend], AuditLog(":memory:"))
        out = []
        for case in load_cases(limit=LIMIT):
            resp = await router.infer(case.request)
            proposed = (resp.action.action == "ADJUST_SETPOINT"
                        or any(a.endswith("physics_rejected") for a in resp.attempts))
            out.append(outcome_from(case, resp.action, proposed_adjust=proposed,
                                    rejected=resp.rejected_reason, warnings=resp.warnings,
                                    tier=resp.tier, latency_ms=resp.latency_ms))
        await backend.aclose()
        return summarize(out)

    s = asyncio.run(run())
    print(f"\nutility summary for {MODEL}: {s}")
    return s


def test_proposes_the_adjustment_when_silica_is_sustained_high(summary):
    assert summary["adjust_recall_pct"] >= MIN_RECALL, summary


def test_does_not_adjust_a_calm_plant(summary):
    assert summary["false_adjust_pct"] <= MAX_FALSE_ADJ, summary


def test_waits_when_the_procedure_says_wait(summary):
    # Sílice alta pero una sola muestra, o ya bajando: el procedimiento dice esperar.
    assert summary["near_miss_false_adjust_pct"] <= MAX_FALSE_ADJ, summary


def test_never_moves_amine_the_wrong_way(summary):
    assert summary["adjust_wrong_direction_pct"] <= MAX_FALSE_ADJ, summary


def test_every_decision_answered(summary):
    assert summary["failsafe_pct"] == 0.0, summary
