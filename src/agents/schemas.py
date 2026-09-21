"""Tipos internos compartidos por el pipeline de agentes.

Deliberadamente independientes de `src.api.routes`: la API tiene su propio
contrato Pydantic (`ChatMessage`, etc.) porque ese es el contrato HTTP externo
(OpenAI-compatible) y puede evolucionar por separado del formato interno que
usan los agentes. `src.api.routes` traduce entre ambos al llamar al
orquestador; ningún módulo de `src.agents` importa nada de `src.api`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class AgentMessage:
    """Mensaje interno del pipeline de agentes (rol + contenido)."""

    role: Role
    content: str


class ToolCallEnvelope(BaseModel):
    """Formato estructurado en el que el SLM solicita la ejecución de una tool."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
