"""AgentOrchestrator: encadena RouterAgent -> AnalyticsAgent (si aplica) -> VerifierAgent.

Nota de diseño sobre `RouterAgent.classify_intent()`: es una capacidad real y
probada de forma independiente (`tests/test_router_agent.py`), pero el
orquestador no la invoca en el camino caliente de cada solicitud, porque
hacerlo agregaría una llamada al LLM adicional por request. En cambio, sí se
usa aquí la parte determinista y sin costo de red del mismo guardrail de
entrada (`RouterAgent.check_threat` / `detect_threat`): bloquea intentos de
inyección de prompt/SQL/comandos antes de tocar el modelo, sin cambiar
cuántas veces se invoca `LLMServer.generate()` por solicitud. Adoptar
`classify_intent` en el camino caliente es un paso natural a futuro, pero
requiere revisar también el contrato de `tests/test_integration.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.engine import GenerationConfig, LLMServer
from src.guardrails.validators import GuardrailError, OutputValidationError, validate_json_output
from src.tools import ToolRegistry

from .analytics_agent import AnalyticsAgent
from .router_agent import RouterAgent
from .schemas import AgentMessage, ToolCallEnvelope
from .verifier_agent import VerifierAgent

TOOL_CALL_INSTRUCTIONS = (
    "Si necesitas una herramienta para responder, contesta ÚNICAMENTE con un objeto JSON "
    'con este formato exacto: {"tool": "<nombre_herramienta>", "arguments": {<argumentos>}}. '
    "Si no necesitas ninguna herramienta, responde directamente en lenguaje natural."
)


class RequestRejectedError(GuardrailError):
    """Se lanza cuando el guardrail de entrada del `RouterAgent` bloquea la solicitud."""


@dataclass
class _Proposal:
    """Turno crudo del SLM: texto y, si aplica, la tool-call que propone."""

    raw_text: str
    tool_call: Optional[ToolCallEnvelope]


@dataclass
class OrchestratorResult:
    """Respuesta final del pipeline, lista para que `src.api.routes` arme el contrato OpenAI."""

    text: str
    used_tool: Optional[str] = None


class AgentOrchestrator:
    """Coordina el pipeline multi-agente: guardrail de entrada -> SLM -> tool (si aplica) -> verificación.

    Un único hop de tool: si tras ejecutar una tool el SLM pide otra, el
    `VerifierAgent` lo rechaza como tool-call sin resolver en vez de
    encadenar indefinidamente.
    """

    def __init__(
        self,
        llm_server: LLMServer,
        tool_registry: ToolRegistry,
        router_agent: Optional[RouterAgent] = None,
        analytics_agent: Optional[AnalyticsAgent] = None,
        verifier_agent: Optional[VerifierAgent] = None,
    ) -> None:
        self.llm_server = llm_server
        self.tool_registry = tool_registry
        self.router_agent = router_agent or RouterAgent(llm_server)
        self.analytics_agent = analytics_agent or AnalyticsAgent(tool_registry)
        self.verifier_agent = verifier_agent or VerifierAgent()

    def _latest_user_text(self, messages: List[AgentMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return ""

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

    def _propose(self, messages: List[AgentMessage], config: GenerationConfig) -> _Proposal:
        raw_text = self.llm_server.generate(self._build_prompt(messages), config=config)
        return _Proposal(raw_text=raw_text, tool_call=self._try_parse_tool_call(raw_text))

    def run(self, messages: List[AgentMessage], config: GenerationConfig) -> OrchestratorResult:
        user_text = self._latest_user_text(messages)
        threat = self.router_agent.check_threat(user_text) if user_text else None
        if threat is not None:
            raise RequestRejectedError(
                f"Solicitud rechazada por el guardrail de entrada (patrón detectado: {threat})."
            )

        proposal = self._propose(messages, config)
        tool_context = None
        used_tool: Optional[str] = None

        if proposal.tool_call is not None:
            used_tool = proposal.tool_call.tool
            analytics_result = self.analytics_agent.execute(proposal.tool_call)
            tool_context = analytics_result.raw_result

            follow_up = [
                *messages,
                AgentMessage(role="assistant", content=proposal.raw_text),
                analytics_result.message,
            ]
            proposal = self._propose(follow_up, config)

        self.verifier_agent.verify(proposal.raw_text, tool_context=tool_context)

        return OrchestratorResult(text=proposal.raw_text, used_tool=used_tool)
