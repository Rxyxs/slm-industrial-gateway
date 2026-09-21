"""RouterAgent: primer eslabón del pipeline.

Construye el prompt a partir de la conversación y el listado de tools
disponibles, invoca al SLM (`LLMServer.generate`) y decide si el texto
devuelto es una respuesta final en lenguaje natural o una solicitud de
ejecución de tool (`{"tool": ..., "arguments": {...}}`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.engine import GenerationConfig, LLMServer
from src.guardrails.validators import OutputValidationError, validate_json_output
from src.tools import ToolRegistry

from .schemas import AgentMessage, ToolCallEnvelope

TOOL_CALL_INSTRUCTIONS = (
    "Si necesitas una herramienta para responder, contesta ÚNICAMENTE con un objeto JSON "
    'con este formato exacto: {"tool": "<nombre_herramienta>", "arguments": {<argumentos>}}. '
    "Si no necesitas ninguna herramienta, responde directamente en lenguaje natural."
)


@dataclass
class RouterDecision:
    """Resultado de un turno de enrutamiento."""

    raw_text: str
    tool_call: Optional[ToolCallEnvelope]


class RouterAgent:
    """Genera el siguiente turno del SLM y enruta entre tool-call y respuesta final."""

    def __init__(self, llm_server: LLMServer, tool_registry: ToolRegistry) -> None:
        self.llm_server = llm_server
        self.tool_registry = tool_registry

    def _build_prompt(self, messages: List[AgentMessage]) -> str:
        tool_names = ", ".join(self.tool_registry.list_tools())
        preamble = f"system: Herramientas disponibles: {tool_names}. {TOOL_CALL_INSTRUCTIONS}"
        turns = [preamble] + [f"{message.role}: {message.content}" for message in messages]
        turns.append("assistant:")
        return "\n".join(turns)

    def _try_parse_tool_call(self, text: str) -> Optional[ToolCallEnvelope]:
        try:
            return validate_json_output(text, ToolCallEnvelope)  # type: ignore[return-value]
        except OutputValidationError:
            return None

    def route(self, messages: List[AgentMessage], config: GenerationConfig) -> RouterDecision:
        prompt = self._build_prompt(messages)
        raw_text = self.llm_server.generate(prompt, config=config)
        return RouterDecision(raw_text=raw_text, tool_call=self._try_parse_tool_call(raw_text))
