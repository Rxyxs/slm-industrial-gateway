"""PdMAgent: agente de mantenimiento predictivo (curvas de falla, RUL y salud de activos).

A diferencia de `RouterAgent`/`AnalyticsAgent`/`VerifierAgent`, no depende del
SLM ni de una decisión del modelo sobre qué tool invocar: recibe directamente
series de condición (vibración, temperatura, carga) de un activo crítico
(molino SAG, camión de extracción, bomba), llama a la tool `calculate_rul` de
`src/tools/` una vez por métrica, e interpreta el resultado crudo en un
diagnóstico de salud con nivel de confianza y ventana de mantenimiento
recomendada. Ejecución 100% in-process (sin red), consistente con el resto
del pipeline air-gapped.

Degrada con gracia por diseño: una métrica con datos insuficientes o fuera de
rango no aborta el diagnóstico del activo completo, solo reduce la confianza
de esa métrica (o la marca `insufficient_data`) y se refleja en el resultado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.tools import ToolExecutionError, ToolNotFoundError, ToolRegistry

CALCULATE_RUL_TOOL = "calculate_rul"

# Rangos físicamente plausibles por tipo de métrica de condición. Una lectura
# fuera de rango no se descarta (puede ser la señal real de una falla), pero
# se marca y reduce la confianza del diagnóstico de esa métrica.
PLAUSIBLE_RANGES: Dict[str, Tuple[float, float]] = {
    "vibration": (0.0, 50.0),  # mm/s RMS
    "temperature": (-40.0, 300.0),  # °C
    "load": (0.0, 150.0),  # % de carga nominal
}

DEFAULT_CRITICAL_RUL_HOURS = 24.0
DEFAULT_WARNING_RUL_HOURS = 168.0


class HealthStatus(str, Enum):
    """Estado de salud de una métrica o de un activo completo."""

    HEALTHY = "healthy"
    DEGRADING = "degrading"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class ConditionSeries(BaseModel):
    """Serie de telemetría de una métrica de condición de un activo."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(..., min_length=1, description="'vibration', 'temperature', 'load' u otra métrica de condición.")
    timestamps: List[float] = Field(default_factory=list, description="Marcas de tiempo en horas, orden creciente.")
    measurements: List[float] = Field(default_factory=list, description="Lecturas del sensor en cada marca de tiempo.")
    failure_threshold: float = Field(..., description="Valor del indicador a partir del cual se considera falla.")

    @model_validator(mode="after")
    def _check_matching_lengths(self) -> "ConditionSeries":
        if len(self.timestamps) != len(self.measurements):
            raise ValueError("'timestamps' y 'measurements' deben tener la misma longitud.")
        return self


@dataclass
class MetricDiagnosis:
    """Diagnóstico de una única métrica de condición."""

    metric: str
    status: HealthStatus
    trend: str
    rul_estimate: Optional[float]
    confidence: float
    has_out_of_range_readings: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class AssetDiagnosis:
    """Diagnóstico agregado de un activo a partir de todas sus métricas de condición."""

    asset_id: str
    overall_status: HealthStatus
    overall_confidence: float
    bottleneck_metric: Optional[str]
    recommended_maintenance_window_hours: Optional[float]
    metric_diagnoses: List[MetricDiagnosis]
    summary: str


class PdMAgent:
    """Interpreta telemetría de condición y produce diagnósticos de salud con RUL estimado."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        critical_rul_hours: float = DEFAULT_CRITICAL_RUL_HOURS,
        warning_rul_hours: float = DEFAULT_WARNING_RUL_HOURS,
    ) -> None:
        if critical_rul_hours < 0 or warning_rul_hours < 0:
            raise ValueError("Los umbrales de RUL deben ser no negativos.")
        if warning_rul_hours < critical_rul_hours:
            raise ValueError("'warning_rul_hours' debe ser mayor o igual que 'critical_rul_hours'.")

        self.tool_registry = tool_registry
        self.critical_rul_hours = critical_rul_hours
        self.warning_rul_hours = warning_rul_hours

    def diagnose(self, asset_id: str, series: List[ConditionSeries]) -> AssetDiagnosis:
        """Diagnostica un activo a partir de una o más series de condición.

        Nunca lanza por datos de sensores incompletos o fuera de rango: cada
        métrica problemática queda reflejada como `INSUFFICIENT_DATA` (o con
        `has_out_of_range_readings=True`) en el resultado en vez de abortar el
        diagnóstico completo. Solo lanza `ValueError` si no se entrega
        ninguna serie.
        """
        if not series:
            raise ValueError(f"Se requiere al menos una serie de condición para diagnosticar '{asset_id}'.")

        metric_diagnoses = [self._diagnose_metric(item) for item in series]
        return self._aggregate(asset_id, metric_diagnoses)

    def _diagnose_metric(self, series: ConditionSeries) -> MetricDiagnosis:
        if len(series.timestamps) < 2:
            return MetricDiagnosis(
                metric=series.metric,
                status=HealthStatus.INSUFFICIENT_DATA,
                trend="unknown",
                rul_estimate=None,
                confidence=0.0,
                has_out_of_range_readings=False,
                notes=["Se requieren al menos 2 lecturas para estimar una tendencia."],
            )

        out_of_range = self._detect_out_of_range(series)
        notes: List[str] = []
        if out_of_range:
            notes.append(
                f"{len(out_of_range)} lectura(s) fuera del rango físicamente plausible para '{series.metric}'."
            )

        try:
            result = self.tool_registry.dispatch(
                CALCULATE_RUL_TOOL,
                {
                    "timestamps": series.timestamps,
                    "measurements": series.measurements,
                    "failure_threshold": series.failure_threshold,
                },
            )
        except (ToolExecutionError, ToolNotFoundError) as exc:
            return MetricDiagnosis(
                metric=series.metric,
                status=HealthStatus.INSUFFICIENT_DATA,
                trend="unknown",
                rul_estimate=None,
                confidence=0.0,
                has_out_of_range_readings=bool(out_of_range),
                notes=notes + [f"No se pudo calcular RUL: {exc}"],
            )

        rul_estimate = result["rul_estimate"]
        confidence = self._estimate_confidence(series, out_of_range)

        return MetricDiagnosis(
            metric=series.metric,
            status=self._classify_status(rul_estimate),
            trend=self._classify_trend(result["slope"]),
            rul_estimate=rul_estimate,
            confidence=confidence,
            has_out_of_range_readings=bool(out_of_range),
            notes=notes,
        )

    def _detect_out_of_range(self, series: ConditionSeries) -> List[int]:
        bounds = PLAUSIBLE_RANGES.get(series.metric)
        if bounds is None:
            return []
        low, high = bounds
        return [index for index, value in enumerate(series.measurements) if value < low or value > high]

    def _estimate_confidence(self, series: ConditionSeries, out_of_range_indices: List[int]) -> float:
        sample_count = len(series.measurements)
        # Más puntos dan más confianza en la tendencia, con rendimientos decrecientes.
        sample_confidence = min(1.0, sample_count / 10.0)
        penalty = len(out_of_range_indices) / sample_count
        return round(max(0.0, sample_confidence - penalty), 3)

    def _classify_trend(self, slope: float, epsilon: float = 1e-9) -> str:
        if abs(slope) < epsilon:
            return "stable"
        return "increasing" if slope > 0 else "decreasing"

    def _classify_status(self, rul_estimate: Optional[float]) -> HealthStatus:
        if rul_estimate is None:
            return HealthStatus.HEALTHY
        if rul_estimate <= self.critical_rul_hours:
            return HealthStatus.CRITICAL
        if rul_estimate <= self.warning_rul_hours:
            return HealthStatus.DEGRADING
        return HealthStatus.HEALTHY

    def _aggregate(self, asset_id: str, metric_diagnoses: List[MetricDiagnosis]) -> AssetDiagnosis:
        valid = [d for d in metric_diagnoses if d.status != HealthStatus.INSUFFICIENT_DATA]

        bottleneck_metric: Optional[str] = None
        min_rul: Optional[float] = None
        finite_ruls = [(d.metric, d.rul_estimate) for d in valid if d.rul_estimate is not None]
        if finite_ruls:
            bottleneck_metric, min_rul = min(finite_ruls, key=lambda item: item[1])

        if not valid:
            overall_status = HealthStatus.INSUFFICIENT_DATA
        elif any(d.status == HealthStatus.CRITICAL for d in valid):
            overall_status = HealthStatus.CRITICAL
        elif any(d.status == HealthStatus.DEGRADING for d in valid):
            overall_status = HealthStatus.DEGRADING
        else:
            overall_status = HealthStatus.HEALTHY

        overall_confidence = round(sum(d.confidence for d in valid) / len(valid), 3) if valid else 0.0

        return AssetDiagnosis(
            asset_id=asset_id,
            overall_status=overall_status,
            overall_confidence=overall_confidence,
            bottleneck_metric=bottleneck_metric,
            recommended_maintenance_window_hours=min_rul,
            metric_diagnoses=metric_diagnoses,
            summary=self._build_summary(asset_id, overall_status, bottleneck_metric, min_rul),
        )

    def _build_summary(
        self,
        asset_id: str,
        overall_status: HealthStatus,
        bottleneck_metric: Optional[str],
        min_rul: Optional[float],
    ) -> str:
        if overall_status == HealthStatus.INSUFFICIENT_DATA:
            return f"Activo '{asset_id}': datos insuficientes para diagnosticar su estado de salud."
        if overall_status == HealthStatus.HEALTHY:
            return f"Activo '{asset_id}': sin tendencia de degradación hacia falla en ninguna métrica monitoreada."

        window = f"{min_rul:.1f} h" if min_rul is not None else "indeterminada"
        urgency = (
            "Se recomienda programar mantenimiento de inmediato"
            if overall_status == HealthStatus.CRITICAL
            else "Se recomienda planificar una ventana de mantenimiento preventivo"
        )
        return f"Activo '{asset_id}': degradación detectada en '{bottleneck_metric}' (RUL estimado: {window}). {urgency}."
