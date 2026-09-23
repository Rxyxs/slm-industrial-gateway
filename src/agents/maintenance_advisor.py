"""MaintenanceAdvisorAgent: capa de negocio ligera, después de que
`AnalyticsAgent` ejecuta `calculate_rul` o `sensor_anomaly_check` dentro del
camino caliente de `/v1/chat/completions`.

No confundir con `PdMAgent` (`pdm_agent.py`): ese es un diagnóstico completo
multi-métrica invocado explícitamente (vía `/v1/pdm/diagnose`) con series de
condición crudas. Este agente, en cambio, vive dentro de `AgentOrchestrator`:
no ejecuta ninguna tool ni recibe series -- solo interpreta el resultado que
`AnalyticsAgent` ya obtuvo durante la conversación (el número de RUL, o los
índices anómalos) contra un umbral de urgencia FIJO, y decide si la respuesta
final debe llevar una alerta de mantenimiento. Mismo patrón que
`SafetyComplianceAgent` aplica sobre límites físicos de diseño (ver
`safety_agent.py`), pero sobre salud de equipo en vez de límites de
seguridad -- y por eso, a diferencia de `SafetyComplianceAgent`, es
deliberadamente **advisor, no bloqueante**: un RUL bajo es información
operativa para planificar mantenimiento, no una condición insegura que deba
impedir que la respuesta llegue al cliente. `AgentOrchestrator` adjunta el
resultado como metadata (`OrchestratorResult.maintenance_alert`) en vez de
lanzar una excepción.

Determinista y sin red: comparación numérica pura, mismo costo que
`SafetyComplianceAgent.audit` en el camino caliente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Umbrales de urgencia -- constantes de módulo, igual que OPERATIONAL_LIMITS
# en safety_agent.py: fijos, no leídos de entorno ni configurables en runtime.
DEFAULT_RUL_URGENT_THRESHOLD_HOURS = 24.0
DEFAULT_ANOMALY_URGENT_COUNT = 2

_RELEVANT_TOOLS = ("calculate_rul", "sensor_anomaly_check")


@dataclass
class MaintenanceAssessment:
    """Resultado de interpretar el resultado crudo de una tool de PdM."""

    is_urgent: bool
    tool: Optional[str]
    reason: Optional[str] = None
    rul_estimate: Optional[float] = None
    anomaly_count: Optional[int] = None


class MaintenanceAdvisorAgent:
    """Interpreta el resultado de `calculate_rul`/`sensor_anomaly_check` y
    decide si amerita una alerta de mantenimiento urgente.

    No ejecuta ninguna tool ni recalcula nada: `assess` recibe el nombre de
    la tool que ya corrió `AnalyticsAgent` y su resultado crudo (el mismo
    `tool_context` que ya recibe `VerifierAgent`).
    """

    def __init__(
        self,
        rul_urgent_threshold_hours: float = DEFAULT_RUL_URGENT_THRESHOLD_HOURS,
        anomaly_urgent_count: int = DEFAULT_ANOMALY_URGENT_COUNT,
    ) -> None:
        self.rul_urgent_threshold_hours = rul_urgent_threshold_hours
        self.anomaly_urgent_count = anomaly_urgent_count

    def assess(self, tool_name: Optional[str], tool_result: Optional[Dict[str, Any]]) -> MaintenanceAssessment:
        """Evalúa el resultado de la última tool ejecutada, si corresponde a PdM.

        Devuelve `MaintenanceAssessment(is_urgent=False, tool=None)` si no se
        ejecutó ninguna tool relevante -- esto nunca lanza una excepción, a
        diferencia de `SafetyComplianceAgent.audit`.
        """
        if tool_name not in _RELEVANT_TOOLS or tool_result is None:
            return MaintenanceAssessment(is_urgent=False, tool=None)

        if tool_name == "calculate_rul":
            return self._assess_rul(tool_result)
        return self._assess_anomaly(tool_result)

    def _assess_rul(self, result: Dict[str, Any]) -> MaintenanceAssessment:
        rul = result.get("rul_estimate")
        if rul is None:
            # Tendencia plana o alejándose del umbral de falla: CalculateRULTool
            # ya decidió que no hay proyección de falla (ver industrial_tools.py).
            return MaintenanceAssessment(is_urgent=False, tool="calculate_rul", rul_estimate=None)

        is_urgent = rul <= self.rul_urgent_threshold_hours
        reason = (
            f"RUL estimado de {rul:.1f}h está en o por debajo del umbral de urgencia "
            f"de {self.rul_urgent_threshold_hours:.1f}h."
            if is_urgent
            else None
        )
        return MaintenanceAssessment(is_urgent=is_urgent, tool="calculate_rul", reason=reason, rul_estimate=float(rul))

    def _assess_anomaly(self, result: Dict[str, Any]) -> MaintenanceAssessment:
        indices = result.get("anomaly_indices") or []
        count = len(indices)
        is_urgent = count >= self.anomaly_urgent_count
        reason = (
            f"Se detectaron {count} lecturas anómalas, en o por encima del umbral de "
            f"{self.anomaly_urgent_count}."
            if is_urgent
            else None
        )
        return MaintenanceAssessment(is_urgent=is_urgent, tool="sensor_anomaly_check", reason=reason, anomaly_count=count)
