"""Herramientas industriales para el agente: SQL analítico, detección de anomalías y cálculo de RUL."""

from src.tools.industrial_tools import (
    BaseIndustrialTool,
    CalculateRULInput,
    CalculateRULTool,
    QueryDuckDBInput,
    QueryDuckDBTool,
    SensorAnomalyCheckInput,
    SensorAnomalyCheckTool,
    build_default_registry,
)
from src.tools.registry import (
    RegisteredTool,
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistry,
)

__all__ = [
    "BaseIndustrialTool",
    "CalculateRULInput",
    "CalculateRULTool",
    "QueryDuckDBInput",
    "QueryDuckDBTool",
    "SensorAnomalyCheckInput",
    "SensorAnomalyCheckTool",
    "build_default_registry",
    "RegisteredTool",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistry",
]
