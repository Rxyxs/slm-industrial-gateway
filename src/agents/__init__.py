"""Pipeline multi-agente: RouterAgent -> AnalyticsAgent (si aplica) -> VerifierAgent."""

from .analytics_agent import AnalyticsAgent, AnalyticsResult
from .orchestrator import AgentOrchestrator, OrchestratorResult
from .router_agent import RouterAgent, RouterDecision
from .schemas import AgentMessage, ToolCallEnvelope
from .verifier_agent import FaithfulnessError, VerificationResult, VerifierAgent

__all__ = [
    "AgentOrchestrator",
    "OrchestratorResult",
    "RouterAgent",
    "RouterDecision",
    "AnalyticsAgent",
    "AnalyticsResult",
    "VerifierAgent",
    "VerificationResult",
    "FaithfulnessError",
    "AgentMessage",
    "ToolCallEnvelope",
]
