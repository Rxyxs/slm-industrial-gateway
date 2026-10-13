"""Contexto operativo para el SLM: tendencias, laboratorio y soft sensor como hechos numerados.

Un modelo de 1-3B no razona bien sobre una tabla de 80 números. Este módulo hace el
trabajo aritmético en código (promedios, pendientes, rachas, holgura hasta el límite) y le
entrega al modelo una lista corta de hechos con identificador, cada uno una frase con el
número exacto. Los identificadores son semánticos y estables (`lab_last`, `soft_sensor`,
`amina_flow`, ...), no posicionales: con `F1, F2, ...` el número de un hecho cambiaba según
qué otros hechos existieran, y qwen2.5:1.5b copiaba la lista de ids de los ejemplos
few-shot aunque en la solicitud real apuntaran a otra cosa. El modelo tiene que citar esos identificadores en
`evidence`; el gateway verifica en código que los hechos citados existen y respaldan la
acción y el motivo (`validation.check_citations`). Así "citar el dato antes de proponer"
deja de ser una instrucción que el modelo puede ignorar.

Fuentes del contexto:
- `HourRecord`: promedio horario de cada tag (lo que entrega un historian, o lo que
  `ContextStore` acumula a partir de la telemetría que pasa por el gateway).
- `LabResult`: análisis de sílice del concentrado, disponible `delay_h` horas después de
  tomada la muestra. Nunca se usa un resultado antes de que la planta lo tendría.
- `SoftSensor`: estimación de la sílice de la hora actual, con su error.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.gateway.soft_sensor import LAB_DELAY_H, SoftSensor
from src.gateway.tags import SILICA_HIGH_PCT, TAG_REGISTRY, WRITABLE_TAGS

TREND_TAGS = ("amina_flow", "starch_flow", "column_01_air_flow", "column_01_level")
RANGE_TAGS = ("ore_pulp_ph", "ore_pulp_density")


@dataclass(frozen=True)
class HourRecord:
    ts: datetime
    values: dict[str, float]


@dataclass(frozen=True)
class LabResult:
    sampled_at: datetime
    silica_pct: float
    delay_h: int = LAB_DELAY_H

    @property
    def available_at(self) -> datetime:
        return self.sampled_at + timedelta(hours=self.delay_h)


@dataclass(frozen=True)
class Fact:
    id: str
    kind: str   # lab_last | lab_prev | lab_streak | soft_sensor | setpoint | range
    tag: str
    value: float
    text: str
    flag: str = ""  # "high", "out_of_range", ... (lo que el hecho afirma)


@dataclass
class PlantContext:
    now: datetime
    hours: int
    facts: list[Fact] = field(default_factory=list)

    def get(self, fid: str) -> Fact | None:
        return next((f for f in self.facts if f.id == fid), None)

    def by_kind(self, kind: str, tag: str | None = None) -> Fact | None:
        return next((f for f in self.facts if f.kind == kind and (tag is None or f.tag == tag)), None)

    @property
    def ids(self) -> list[str]:
        return [f.id for f in self.facts]

    def render(self) -> str:
        return "\n".join(f"[{f.id}] {f.text}" for f in self.facts)


def _slope_per_h(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2
    ym = sum(values) / n
    num = sum((i - xm) * (v - ym) for i, v in enumerate(values))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den


def lab_as_of(labs: list[LabResult], ts: datetime) -> LabResult | None:
    """Último resultado cuya muestra se tomó en o antes de `ts`."""
    best = None
    for lab in labs:
        if lab.sampled_at <= ts and (best is None or lab.sampled_at > best.sampled_at):
            best = lab
    return best


def soft_sensor_inputs(now: datetime, current: dict[str, float], labs: list[LabResult],
                       sensor: SoftSensor) -> dict[str, float]:
    x = dict(current)
    for name in sensor.features:
        if name.startswith("lab_lag"):
            lag = int(name.removeprefix("lab_lag"))
            lab = lab_as_of(labs, now - timedelta(hours=lag))
            if lab is not None:
                x[name] = lab.silica_pct
    return x


SILICA_RECOVERY_PTS = 0.3


def silica_state(last: float, prev: float) -> str:
    """Paso 1 del procedimiento, calculado en código."""
    if last >= SILICA_HIGH_PCT and prev >= SILICA_HIGH_PCT:
        return "RECOVERING" if last < prev - SILICA_RECOVERY_PTS else "SUSTAINED_HIGH"
    if last >= SILICA_HIGH_PCT:
        return "SINGLE_HIGH"
    return "NORMAL"


_STATE_TEXT = {
    "SUSTAINED_HIGH": "latest and previous lab results both >= {thr}% and not falling by more than 0.3",
    "RECOVERING": "both results high but the latest fell by more than 0.3 points: already recovering, wait",
    "SINGLE_HIGH": "only the latest lab result is >= {thr}%: not sustained yet, wait for the next one",
    "NORMAL": "latest lab result below {thr}%",
}


def build_context(window: list[HourRecord], labs: list[LabResult],
                  sensor: SoftSensor | None, derived: bool = False) -> PlantContext:
    """`window`: horas consecutivas, la última es la hora actual.

    `derived=True` agrega el hecho `silica_state` (paso 1 del procedimiento ya resuelto en
    código). Es la variante E de la evaluación de utilidad."""
    now = window[-1].ts
    current = window[-1].values
    ctx = PlantContext(now=now, hours=len(window))
    facts: list[Fact] = []

    def add(kind, tag, value, text, flag=""):
        fid = tag if kind in {"setpoint", "range"} else kind
        facts.append(Fact(fid, kind, tag, value, text, flag))

    known = sorted((l for l in labs if l.available_at <= now), key=lambda l: l.sampled_at)
    if known:
        last = known[-1]
        age = (now - last.sampled_at).total_seconds() / 3600
        add("lab_last", "silica_concentrate_pct", last.silica_pct,
            f"latest lab silica in concentrate = {last.silica_pct:.2f}% (sample taken {age:.0f} h ago; "
            f"high means >= {SILICA_HIGH_PCT}%)", "high" if last.silica_pct >= SILICA_HIGH_PCT else "normal")
        if len(known) >= 2:
            prev = known[-2]
            add("lab_prev", "silica_concentrate_pct", prev.silica_pct,
                f"previous lab silica = {prev.silica_pct:.2f}% (change since then "
                f"{last.silica_pct - prev.silica_pct:+.2f} points)",
                "high" if prev.silica_pct >= SILICA_HIGH_PCT else "normal")
        streak = 0
        for lab in reversed(known):
            if lab.silica_pct < SILICA_HIGH_PCT:
                break
            streak += 1
        add("lab_streak", "silica_concentrate_pct", float(streak),
            f"consecutive lab results at or above {SILICA_HIGH_PCT}% = {streak}",
            "sustained_high" if streak >= 2 else "")
        if derived and len(known) >= 2:
            state = silica_state(last.silica_pct, known[-2].silica_pct)
            add("silica_state", "silica_concentrate_pct", last.silica_pct,
                f"silica state (procedure step 1) = {state}: "
                + _STATE_TEXT[state].format(thr=SILICA_HIGH_PCT),
                "sustained_high" if state == "SUSTAINED_HIGH" else
                ("high" if state in {"SINGLE_HIGH", "RECOVERING"} else "normal"))

    if sensor is not None:
        est = sensor.predict(soft_sensor_inputs(now, current, labs, sensor))
        if est is not None:
            add("soft_sensor", "silica_concentrate_pct", est,
                f"soft-sensor estimate of silica now = {est:.2f}% (typical error +/-{sensor.rmse:.2f})",
                "high" if est >= SILICA_HIGH_PCT else "normal")

    for tag in TREND_TAGS:
        if tag not in current:
            continue
        spec = TAG_REGISTRY[tag]
        series = [r.values[tag] for r in window if tag in r.values]
        cur = current[tag]
        slope = _slope_per_h(series)
        headroom = spec.high - cur
        add("setpoint", tag, cur,
            f"{tag} now = {cur:.4g} {spec.unit}; {len(series)} h mean {sum(series) / len(series):.4g}, "
            f"trend {slope:+.3g}/h; allowed [{spec.low:g}, {spec.high:g}], "
            f"room to raise {headroom:.4g}, max change per step {spec.max_step:g}")

    for tag in RANGE_TAGS:
        if tag not in current:
            continue
        spec = TAG_REGISTRY[tag]
        cur = current[tag]
        ok = spec.low <= cur <= spec.high
        add("range", tag, cur, f"{tag} = {cur:.4g} {spec.unit} ({'inside' if ok else 'OUTSIDE'} "
            f"normal range [{spec.low:g}, {spec.high:g}])", "" if ok else "out_of_range")

    ctx.facts = facts
    return ctx


def adjust_allowed(ctx: PlantContext) -> bool:
    """Compuerta del procedimiento: ajustar solo con sílice sostenidamente alta y pH/densidad en rango.

    Medido con qwen2.5:1.5b: aun recibiendo `silica_state = SINGLE_HIGH ... wait`, el modelo
    ajustaba en el 100% de esas horas. Lo que es regla se decide acá, en código; el modelo
    elige dentro de lo que la regla permite.
    """
    state = ctx.by_kind("silica_state")
    if state is None or "SUSTAINED_HIGH" not in state.text:
        return False
    return not any(f.kind == "range" and f.flag == "out_of_range" for f in ctx.facts)


def procedure_bounds(ctx: PlantContext, bounds: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Lo que el procedimiento permite escribir: solo subir amina, y solo si `adjust_allowed`.

    Medido con qwen2.5:3b: con contexto y ±max_step disponibles, 23 de 40 ajustes ejecutados
    *bajaban* la amina, lo contrario de lo que pide el procedimiento. La dirección también
    es regla, así que también se impone en la gramática.
    """
    if not adjust_allowed(ctx) or "amina_flow" not in bounds:
        return {}
    lo, hi = bounds["amina_flow"]
    amina = ctx.by_kind("setpoint", "amina_flow")
    lo = max(lo, math.floor(amina.value) + 1)
    return {"amina_flow": (lo, hi)} if lo <= hi else {}


class ContextStore:
    """Historia por activo dentro del propio gateway.

    Cada lectura que pasa por `/v1/infer` o MQTT se acumula en la hora que le corresponde;
    al cerrar la hora se guarda el promedio. Los resultados de laboratorio llegan por
    `POST /v1/lab` o por `plant/<asset>/lab`. Sin historian externo el gateway igual
    puede armar tendencias, a partir de lo que ya ve pasar.
    """

    def __init__(self, window_h: int = 8, max_labs: int = 48):
        self.window_h = window_h
        self._hours: dict[str, deque[HourRecord]] = defaultdict(lambda: deque(maxlen=window_h))
        self._open: dict[str, tuple[datetime, dict[str, list[float]]]] = {}
        self._labs: dict[str, deque[LabResult]] = defaultdict(lambda: deque(maxlen=max_labs))

    def add_hour(self, asset: str, record: HourRecord) -> None:
        self._hours[asset].append(record)

    def observe(self, asset: str, ts: datetime, values: dict[str, float]) -> None:
        hour = ts.replace(minute=0, second=0, microsecond=0)
        open_hour = self._open.get(asset)
        if open_hour is not None and open_hour[0] != hour:
            h, acc = open_hour
            self.add_hour(asset, HourRecord(h, {t: sum(v) / len(v) for t, v in acc.items()}))
            open_hour = None
        if open_hour is None:
            open_hour = (hour, defaultdict(list))
            self._open[asset] = open_hour
        for tag, v in values.items():
            open_hour[1][tag].append(v)

    def add_lab(self, asset: str, lab: LabResult) -> None:
        self._labs[asset].append(lab)

    def window(self, asset: str) -> list[HourRecord]:
        return list(self._hours[asset])

    def labs(self, asset: str) -> list[LabResult]:
        return list(self._labs[asset])


__all__ = ["ContextStore", "Fact", "HourRecord", "LabResult", "PlantContext", "build_context",
           "WRITABLE_TAGS"]
