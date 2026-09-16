"""Herramientas industriales: consultas analíticas, detección de anomalías y cálculo de RUL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Type

import duckdb
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.guardrails.validators import validate_sql_query


class BaseIndustrialTool(BaseModel, ABC):
    """Clase base para herramientas industriales invocables por el agente."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: ClassVar[str]
    description: ClassVar[str]
    args_schema: ClassVar[Type[BaseModel]]

    @abstractmethod
    def run(self, params: BaseModel) -> Dict[str, Any]:
        """Ejecuta la herramienta con los argumentos ya validados por `args_schema`."""

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        validated = self.args_schema.model_validate(kwargs)
        return self.run(validated)


# --------------------------------------------------------------------------- #
# QueryDuckDBTool
# --------------------------------------------------------------------------- #


class QueryDuckDBInput(BaseModel):
    """Argumentos para una consulta analítica de solo lectura sobre DuckDB."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Consulta SQL de solo lectura (SELECT/WITH) a ejecutar.")
    parameters: Optional[List[Any]] = Field(
        default=None, description="Parámetros posicionales para los marcadores '?' de la consulta."
    )
    limit: int = Field(default=1000, ge=1, le=100_000, description="Número máximo de filas a devolver.")


class QueryDuckDBTool(BaseIndustrialTool):
    """Ejecuta consultas SQL analíticas de solo lectura sobre una base de datos DuckDB."""

    name: ClassVar[str] = "query_duckdb"
    description: ClassVar[str] = (
        "Ejecuta consultas SQL analíticas de solo lectura (SELECT) sobre una base de datos DuckDB "
        "y devuelve columnas y filas resultantes."
    )
    args_schema: ClassVar[Type[BaseModel]] = QueryDuckDBInput

    database: str = ":memory:"

    def run(self, params: QueryDuckDBInput) -> Dict[str, Any]:
        safe_query = validate_sql_query(params.query)

        connection = duckdb.connect(database=self.database)
        try:
            cursor = connection.execute(safe_query, params.parameters or [])
            columns = [column[0] for column in cursor.description] if cursor.description else []
            rows = [list(row) for row in cursor.fetchmany(params.limit)]
            return {"columns": columns, "rows": rows, "row_count": len(rows)}
        finally:
            connection.close()


# --------------------------------------------------------------------------- #
# SensorAnomalyCheckTool
# --------------------------------------------------------------------------- #


class SensorAnomalyCheckInput(BaseModel):
    """Argumentos para la detección de anomalías por Z-score en una serie de lecturas."""

    model_config = ConfigDict(extra="forbid")

    readings: List[float] = Field(
        ..., min_length=2, description="Serie histórica de lecturas, con el valor más reciente al final."
    )
    threshold: float = Field(
        default=3.0, gt=0, description="Umbral de |Z-score| a partir del cual una lectura se considera anómala."
    )


class SensorAnomalyCheckTool(BaseIndustrialTool):
    """Detecta lecturas anómalas de un sensor industrial mediante el método de Z-score."""

    name: ClassVar[str] = "sensor_anomaly_check"
    description: ClassVar[str] = (
        "Analiza una serie de lecturas de un sensor y detecta valores anómalos usando Z-score."
    )
    args_schema: ClassVar[Type[BaseModel]] = SensorAnomalyCheckInput

    def run(self, params: SensorAnomalyCheckInput) -> Dict[str, Any]:
        readings = np.asarray(params.readings, dtype=float)
        mean = float(readings.mean())
        std_dev = float(readings.std(ddof=0))

        # Serie constante: ninguna lectura se desvía de la media.
        z_scores = np.zeros_like(readings) if std_dev == 0.0 else (readings - mean) / std_dev

        anomaly_mask = np.abs(z_scores) >= params.threshold
        anomaly_indices = [int(index) for index in np.flatnonzero(anomaly_mask)]

        return {
            "mean": mean,
            "std_dev": std_dev,
            "z_scores": [float(z) for z in z_scores],
            "anomaly_indices": anomaly_indices,
            "is_last_reading_anomalous": bool(anomaly_mask[-1]),
        }


# --------------------------------------------------------------------------- #
# CalculateRULTool
# --------------------------------------------------------------------------- #


class CalculateRULInput(BaseModel):
    """Argumentos para estimar la vida útil restante (RUL) por extrapolación lineal."""

    model_config = ConfigDict(extra="forbid")

    timestamps: List[float] = Field(
        ..., min_length=2, description="Marcas de tiempo de cada medición, en orden creciente."
    )
    measurements: List[float] = Field(
        ..., min_length=2, description="Valor del indicador de degradación en cada marca de tiempo."
    )
    failure_threshold: float = Field(
        ..., description="Valor del indicador a partir del cual se considera que el equipo falla."
    )

    @model_validator(mode="after")
    def _check_matching_lengths(self) -> "CalculateRULInput":
        if len(self.timestamps) != len(self.measurements):
            raise ValueError("'timestamps' y 'measurements' deben tener la misma longitud.")
        return self


class CalculateRULTool(BaseIndustrialTool):
    """Estima la vida útil restante (RUL) de un equipo por extrapolación lineal de su degradación."""

    name: ClassVar[str] = "calculate_rul"
    description: ClassVar[str] = (
        "Calcula la vida útil restante (Remaining Useful Life) de un equipo ajustando una "
        "tendencia lineal a su indicador de degradación y extrapolando hasta el umbral de falla."
    )
    args_schema: ClassVar[Type[BaseModel]] = CalculateRULInput

    def run(self, params: CalculateRULInput) -> Dict[str, Any]:
        timestamps = np.asarray(params.timestamps, dtype=float)
        measurements = np.asarray(params.measurements, dtype=float)

        slope, intercept = np.polyfit(timestamps, measurements, 1)
        current_time = float(timestamps[-1])
        current_value = float(measurements[-1])
        delta_needed = params.failure_threshold - current_value

        if delta_needed == 0.0:
            rul_estimate: Optional[float] = 0.0
        elif slope == 0.0 or (delta_needed > 0) != (slope > 0):
            # Tendencia plana o alejándose del umbral: no se proyecta una falla.
            rul_estimate = None
        else:
            time_at_threshold = (params.failure_threshold - intercept) / slope
            rul_estimate = max(0.0, time_at_threshold - current_time)

        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "current_value": current_value,
            "rul_estimate": rul_estimate,
            "is_degrading_toward_failure": rul_estimate is not None,
        }


def build_default_registry() -> "ToolRegistry":
    """Crea un `ToolRegistry` con las tres herramientas industriales ya registradas."""
    from src.tools.registry import ToolRegistry

    registry = ToolRegistry()
    for tool in (QueryDuckDBTool(), SensorAnomalyCheckTool(), CalculateRULTool()):
        registry.register_tool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            func=tool.run,
        )
    return registry
