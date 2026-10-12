"""Registro de tags de planta: qué puede leer el gateway y qué puede escribir.

Los límites salen de datos reales, no de un manual inventado: son los percentiles
1 y 99 de cada variable en el dataset público de una planta de flotación de hierro
(Kaggle, "Quality Prediction in a Mining Process", 737.453 filas, marzo-sept 2017).
`max_step` es el percentil 99 del cambio absoluto entre registros consecutivos,
redondeado hacia abajo: un cambio de setpoint mayor que lo que la propia planta
hizo el 99% del tiempo se considera fuera de operación normal y se rechaza.

Un tag que no está acá no existe para el gateway. Si el modelo devuelve un tag
desconocido, eso es una alucinación y la acción se descarta.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TagSpec:
    tag: str
    unit: str
    low: float
    high: float
    writable: bool = False
    max_step: float | None = None
    source_column: str = ""
    # Direcciones en los protocolos de campo (simuladas en los tests).
    opcua_node: str = ""
    modbus_register: int | None = None
    modbus_scale: float = 1.0


TAG_REGISTRY: dict[str, TagSpec] = {
    spec.tag: spec
    for spec in [
        TagSpec("starch_flow", "m3/h", 176.5, 5508.5, writable=True, max_step=300.0,
                source_column="Starch Flow", opcua_node="ns=2;s=Flotation.StarchFlow.SP",
                modbus_register=40001, modbus_scale=1.0),
        TagSpec("amina_flow", "m3/h", 248.8, 671.8, writable=True, max_step=25.0,
                source_column="Amina Flow", opcua_node="ns=2;s=Flotation.AminaFlow.SP",
                modbus_register=40002, modbus_scale=10.0),
        TagSpec("column_01_air_flow", "Nm3/h", 177.4, 303.0, writable=True, max_step=5.0,
                source_column="Flotation Column 01 Air Flow",
                opcua_node="ns=2;s=Flotation.Col01.AirFlow.SP", modbus_register=40003,
                modbus_scale=10.0),
        TagSpec("column_01_level", "mm", 315.3, 854.4, writable=True, max_step=20.0,
                source_column="Flotation Column 01 Level",
                opcua_node="ns=2;s=Flotation.Col01.Level.SP", modbus_register=40004,
                modbus_scale=1.0),
        TagSpec("ore_pulp_flow", "t/h", 377.0, 417.8, source_column="Ore Pulp Flow"),
        TagSpec("ore_pulp_ph", "pH", 8.75, 10.6, source_column="Ore Pulp pH"),
        TagSpec("ore_pulp_density", "kg/cm3", 1.52, 1.79, source_column="Ore Pulp Density"),
        TagSpec("iron_feed_pct", "%", 45.38, 64.03, source_column="% Iron Feed"),
        TagSpec("silica_feed_pct", "%", 6.08, 30.14, source_column="% Silica Feed"),
        TagSpec("silica_concentrate_pct", "%", 0.98, 5.27,
                source_column="% Silica Concentrate"),
    ]
}

# Sílice en concentrado "alta": percentil 90 del promedio horario (4.097 horas).
SILICA_HIGH_PCT = 4.14

WRITABLE_TAGS: tuple[str, ...] = tuple(t for t, s in TAG_REGISTRY.items() if s.writable)
