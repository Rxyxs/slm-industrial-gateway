"""Pipeline multi-agente: RouterAgent (guardrail de entrada) -> AnalyticsAgent
(si aplica, con evaluación MaintenanceAdvisorAgent) -> VerifierAgent ->
SafetyComplianceAgent (bloqueante), coordinados por AgentOrchestrator.

`PdMAgent` es una capacidad aparte: diagnóstico multi-métrica invocado
explícitamente vía `/v1/pdm/diagnose`, no parte de ese pipeline de chat."""

from .analytics_agent import AnalyticsAgent, AnalyticsResult
from .maintenance_advisor import MaintenanceAdvisorAgent, MaintenanceAssessment
from .orchestrator import AgentOrchestrator, OrchestratorResult, RequestRejectedError
from .pdm_agent import AssetDiagnosis, ConditionSeries, HealthStatus, MetricDiagnosis, PdMAgent
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
    "MaintenanceAdvisorAgent",
    "MaintenanceAssessment",
    "PdMAgent",
    "ConditionSeries",
    "HealthStatus",
    "MetricDiagnosis",
    "AssetDiagnosis",
    "SafetyComplianceAgent",
    "SafetyCheckResult",
    "SafetyAlertError",
    "AgentMessage",
    "ToolCallEnvelope",
]
