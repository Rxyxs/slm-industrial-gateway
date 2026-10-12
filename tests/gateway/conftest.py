from __future__ import annotations

import asyncio
import json

import pytest

from src.gateway.audit import AuditLog
from src.gateway.backends import BackendError
from src.gateway.config import GatewaySettings
from src.gateway.schemas import InferenceRequest


class FakeBackend:
    """Backend programable: cada llamada consume el siguiente elemento del guion.
    Un str se devuelve como salida del modelo, una excepción se lanza, un float
    es una espera en segundos (para simular un motor lento)."""

    def __init__(self, name: str, tier: str = "edge", script=None, max_complexity: float = 1.0,
                 default=None):
        self.name, self.tier, self.max_complexity = name, tier, max_complexity
        self.script = list(script or [])
        self.default = default
        self.calls = 0
        self.last_user_prompt = ""

    async def generate(self, system, user, schema, timeout_s):
        self.calls += 1
        self.last_user_prompt = user
        step = self.script.pop(0) if self.script else self.default
        if isinstance(step, float):
            await asyncio.sleep(step)
            step = self.script.pop(0) if self.script else self.default
        if isinstance(step, Exception):
            raise step
        if step is None:
            raise BackendError(f"{self.name}: no script")
        return step

    async def aclose(self):
        pass


def action_json(**overrides) -> str:
    base = {"action": "HOLD", "reason_code": "WITHIN_NORMAL", "confidence": "HIGH",
            "rationale": "all readings inside normal range"}
    base.update(overrides)
    return json.dumps(base)


ADJUST_AMINA = action_json(action="ADJUST_SETPOINT", target_tag="amina_flow", new_value=520.0,
                           reason_code="SILICA_HIGH", confidence="MEDIUM",
                           rationale="raise amine to float more silica")


@pytest.fixture
def plant_request() -> InferenceRequest:
    # Valores de la primera fila del dataset real de flotación (2017-03-10 01:00).
    return InferenceRequest(
        asset_id="flot-col-01",
        telemetry=[
            {"tag": "starch_flow", "value": 3019.53},
            {"tag": "amina_flow", "value": 557.434},
            {"tag": "ore_pulp_ph", "value": 10.0664},
            {"tag": "ore_pulp_density", "value": 1.74},
            {"tag": "column_01_air_flow", "value": 249.214},
            {"tag": "column_01_level", "value": 457.396},
        ],
    )


@pytest.fixture
def settings() -> GatewaySettings:
    return GatewaySettings(deadline_s=1.0, allow_cloud=True, breaker_failure_threshold=2,
                           breaker_reset_timeout_s=30.0, audit_db_path=":memory:")


@pytest.fixture
def audit():
    log = AuditLog(":memory:")
    yield log
    log.close()
