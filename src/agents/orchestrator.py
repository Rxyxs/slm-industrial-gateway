"""AgentOrchestrator: encadena RouterAgent -> AnalyticsAgent (si aplica) -> VerifierAgent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.engine import GenerationConfig, LLMServer
from src.tools import ToolRegistry

from .analytics_agent import AnalyticsAgent
from .router_agent import RouterAgent
from .schemas import AgentMessage
from .verifier_agent import VerifierAgent


@dataclass
class OrchestratorResult:
    """Respuesta final del pipeline, lista para que `src.api.routes` arme el contrato OpenAI."""

    text: str
    used_tool: Optional[str] = None


class AgentOrchestrator:
    """Coordina el pipeline multi-agente: enrutamiento -> ejecución de tool -> verificación.

    Un único hop de tool: si tras ejecutar una tool el SLM pide otra, el
    `VerifierAgent` lo rechaza como tool-call sin resolver en vez de
    encadenar indefinidamente (mismo alcance que el flujo original de
    `src.api.routes`, ahora repartido en agentes independientes).
    """

    def __init__(
        self,
        llm_server: LLMServer,
        tool_registry: ToolRegistry,
        router_agent: Optional[RouterAgent] = None,
        analytics_agent: Optional[AnalyticsAgent] = None,
        verifier_agent: Optional[VerifierAgent] = None,
    ) -> None:
        self.router_agent = router_agent or RouterAgent(llm_server, tool_registry)
        self.analytics_agent = analytics_agent or AnalyticsAgent(tool_registry)
        self.verifier_agent = verifier_agent or VerifierAgent()

    def run(self, messages: List[AgentMessage], config: GenerationConfig) -> OrchestratorResult:
        decision = self.router_agent.route(messages, config)
        tool_context = None
        used_tool: Optional[str] = None

        if decision.tool_call is not None:
            used_tool = decision.tool_call.tool
            analytics_result = self.analytics_agent.execute(decision.tool_call)
            tool_context = analytics_result.raw_result

            follow_up = [
                *messages,
                AgentMessage(role="assistant", content=decision.raw_text),
                analytics_result.message,
            ]
            decision = self.router_agent.route(follow_up, config)

        self.verifier_agent.verify(decision.raw_text, tool_context=tool_context)

        return OrchestratorResult(text=decision.raw_text, used_tool=used_tool)
