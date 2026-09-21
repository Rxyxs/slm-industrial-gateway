"""AnalyticsAgent: segundo eslabón del pipeline.

Despacha la tool solicitada por el `RouterAgent` contra el `ToolRegistry` real
(DuckDB, detección de anomalías, cálculo de RUL) y empaqueta el resultado como
un mensaje de rol `tool` listo para reinyectar al SLM, además de conservar el
resultado crudo para que el `VerifierAgent` pueda usarlo como contexto de
verificación.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.tools import ToolRegistry

from .schemas import AgentMessage, ToolCallEnvelope


def _json_default(value: Any) -> Any:
    """Serializa tipos que las tools (p. ej. DuckDB) devuelven y que `json` no soporta nativamente."""
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass
class AnalyticsResult:
    """Resultado de ejecutar una tool: el mensaje a reinyectar al SLM y el resultado crudo."""

    message: AgentMessage
    raw_result: Any


class AnalyticsAgent:
    """Ejecuta herramientas analíticas industriales a través del `ToolRegistry`."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry

    def execute(self, tool_call: ToolCallEnvelope) -> AnalyticsResult:
        """Despacha `tool_call` (puede lanzar `ToolNotFoundError`/`ToolExecutionError`/`GuardrailError`)."""
        raw_result = self.tool_registry.dispatch(tool_call.tool, tool_call.arguments)
        content = json.dumps(raw_result, ensure_ascii=False, default=_json_default)
        return AnalyticsResult(message=AgentMessage(role="tool", content=content), raw_result=raw_result)
