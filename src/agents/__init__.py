"""Pipeline multi-agente: RouterAgent (guardrail de entrada) -> AnalyticsAgent
(si aplica) -> VerifierAgent, coordinados por AgentOrchestrator."""

from .analytics_agent import AnalyticsAgent, AnalyticsResult
from .orchestrator import AgentOrchestrator, OrchestratorResult, RequestRejectedError
from .router_agent import RequestIntent, RouterAgent, detect_threat, sanitize_input
from .schemas import AgentMessage, ToolCallEnvelope
from .verifier_agent import FaithfulnessError, VerificationResult, VerifierAgent

__all__ = [
    "AgentOrchestrator",
    "OrchestratorResult",
    "RequestRejectedError",
    "RouterAgent",
    "RequestIntent",
    "detect_threat",
    "sanitize_input",
    "AnalyticsAgent",
    "AnalyticsResult",
    "VerifierAgent",
    "VerificationResult",
    "FaithfulnessError",
    "AgentMessage",
    "ToolCallEnvelope",
]
