"""Pipeline multi-agente: RouterAgent (guardrail de entrada) -> AnalyticsAgent
(si aplica, con evaluación PdMAgent advisor) -> VerifierAgent ->
SafetyComplianceAgent (bloqueante), coordinados por AgentOrchestrator."""

from .analytics_agent import AnalyticsAgent, AnalyticsResult
from .orchestrator import AgentOrchestrator, OrchestratorResult, RequestRejectedError
from .pdm_agent import MaintenanceAssessment, PdMAgent
from .router_agent import RequestIntent, RouterAgent, detect_threat, sanitize_input
from .safety_agent import SafetyAlertError, SafetyCheckResult, SafetyComplianceAgent
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
    "PdMAgent",
    "MaintenanceAssessment",
    "SafetyComplianceAgent",
    "SafetyCheckResult",
    "SafetyAlertError",
    "AgentMessage",
    "ToolCallEnvelope",
]
