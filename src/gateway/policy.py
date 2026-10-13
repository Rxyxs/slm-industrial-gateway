"""Regla operativa de referencia para las evaluaciones de utilidad.

El dataset no dice qué setpoint era el correcto: en flotation-plant-silica-soft-sensor
cuatro modelos entrenados en distintos tramos de historia recomiendan mover los controles
en direcciones opuestas. Lo que sí está respaldado es la práctica de la propia planta:

- En flotación catiónica inversa de hierro la amina es el colector que flota el cuarzo;
  más amina baja la sílice del concentrado.
- Los operadores de esta planta suben la amina cuando la sílice medida sube (la dosis
  correlaciona 0,23 con la sílice de dos horas antes, más que con la de la misma hora).

La regla codifica esa práctica, no un óptimo. Una hora es:
- ADJUST (subir amina) si la sílice está alta de forma sostenida y nada la contradice:
  los dos últimos resultados de laboratorio >= 4,14% (p90 horario), el último no viene
  bajando más de 0,3 puntos, el soft sensor no lo contradice (estima a menos de un error
  típico del umbral: >= 4,14 - RMSE), pH y densidad en rango, y hay al menos 10 m3/h de
  holgura de amina. No se exige que el soft sensor marque >= 4,14: el Ridge se encoge
  hacia la media y en las horas de prueba no lo hizo nunca, aunque el laboratorio
  mostrara sílice alta dos veces seguidas en 67 horas.
- HOLD si todo está tranquilo: los dos últimos resultados y el soft sensor <= 2,5% y pH y
  densidad en rango.
- HOLD_NEAR_MISS si la sílice está alta pero el procedimiento igual dice esperar: el último
  resultado >= 4,14% y el anterior < 4,14% (una sola muestra alta), o los dos altos pero
  el último bajando más de 0,3 puntos (ya se está recuperando). Son los casos que separan
  a un modelo que sigue el procedimiento de uno que aprendió "sílice alta -> ajustar".
- Ninguna (None): la hora es ambigua y no entra a la evaluación.
"""

from __future__ import annotations

from src.gateway.context import PlantContext
from src.gateway.tags import SILICA_HIGH_PCT, TAG_REGISTRY

CALM_PCT = 2.5
MIN_HEADROOM = 10.0
MAX_RECOVERY = 0.3


def label_case(ctx: PlantContext, sensor_rmse: float) -> str | None:
    last, prev = ctx.by_kind("lab_last"), ctx.by_kind("lab_prev")
    soft = ctx.by_kind("soft_sensor")
    amina = ctx.by_kind("setpoint", "amina_flow")
    ranges = [f for f in ctx.facts if f.kind == "range"]
    if None in (last, prev, soft, amina) or len(ranges) < 2:
        return None
    if any(f.flag == "out_of_range" for f in ranges):
        return None
    if (last.value >= SILICA_HIGH_PCT and prev.value >= SILICA_HIGH_PCT
            and last.value >= prev.value - MAX_RECOVERY and soft.value >= SILICA_HIGH_PCT - sensor_rmse
            and TAG_REGISTRY["amina_flow"].high - amina.value >= MIN_HEADROOM):
        return "ADJUST"
    if last.value <= CALM_PCT and prev.value <= CALM_PCT and soft.value <= CALM_PCT:
        return "HOLD"
    if last.value >= SILICA_HIGH_PCT and (
            prev.value < SILICA_HIGH_PCT or last.value < prev.value - MAX_RECOVERY):
        return "HOLD_NEAR_MISS"
    return None
