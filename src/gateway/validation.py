"""Validación en dos capas de lo que devuelve el modelo.

1. Forma: el texto tiene que ser exactamente un objeto JSON que cumpla
   `ModelOutput`. No se "rescata" JSON de entre texto libre: si el motor no
   respetó el schema, la respuesta no es confiable y se descarta. La única
   corrección permitida es recortar `rationale`, que no es ejecutable.
2. Física: un setpoint tiene que caer dentro de los límites del tag y no puede
   moverse más que `max_step` respecto del valor actual medido. Sin valor actual
   en la telemetría no hay forma de acotar el paso, así que también se rechaza.

Cualquier rechazo se convierte en HOLD (no tocar nada). En control de planta el
modo de falla seguro es dejar el proceso como está y avisar, no adivinar.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from src.gateway.schemas import (
    MODEL_OUTPUT, ActionType, Confidence, ControlAction, ReasonCode, TelemetrySample,
)
from src.gateway.tags import SILICA_HIGH_PCT, TAG_REGISTRY


class OutputRejected(Exception):
    """La salida del modelo no es ejecutable."""


def parse_action(raw: str) -> ControlAction:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OutputRejected(f"not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise OutputRejected("top-level JSON must be an object")
    try:
        return ControlAction.from_output(MODEL_OUTPUT.validate_python(payload))
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "<root>"
        raise OutputRejected(f"schema violation at {loc}: {first['msg']}") from exc


def check_physics(action: ControlAction, telemetry: list[TelemetrySample]) -> None:
    if action.action != ActionType.ADJUST_SETPOINT.value:
        return
    spec = TAG_REGISTRY[action.target_tag]
    value = float(action.new_value)
    if not (spec.low <= value <= spec.high):
        raise OutputRejected(
            f"{spec.tag}={value:g} outside [{spec.low:g}, {spec.high:g}] {spec.unit}"
        )
    current = next(
        (s for s in telemetry if s.tag == spec.tag and s.quality == "good"), None
    )
    if current is None:
        raise OutputRejected(f"no good-quality reading of {spec.tag} to bound the step")
    step = abs(value - current.value)
    if spec.max_step is not None and step > spec.max_step:
        raise OutputRejected(
            f"{spec.tag} step {step:g} exceeds max_step {spec.max_step:g} {spec.unit}"
        )


_RANGE_PREMISES = {
    "PH_OUT_OF_RANGE": ("ore_pulp_ph",),
    "DENSITY_OUT_OF_RANGE": ("ore_pulp_density",),
    "LEVEL_DEVIATION": ("column_01_level",),
    "AIR_FLOW_DEVIATION": ("column_01_air_flow",),
}


def check_premise(action: ControlAction, telemetry: list[TelemetrySample]) -> str | None:
    """¿Lo que el modelo dice que motivó la acción es cierto en la telemetría?

    Devuelve un aviso si `reason_code` afirma algo que los datos contradicen
    (p. ej. PH_OUT_OF_RANGE con el pH dentro de rango). Medido con
    qwen2.5:1.5b: la explicación libre llegaba a contradecir la telemetría, así
    que el motivo estructurado se verifica en código en vez de confiar en él.
    """
    by_tag = {s.tag: s for s in telemetry}
    code = action.reason_code
    if code in _RANGE_PREMISES:
        tags = _RANGE_PREMISES[code]
        readings = [by_tag[t] for t in tags if t in by_tag]
        if readings and all(TAG_REGISTRY[r.tag].low <= r.value <= TAG_REGISTRY[r.tag].high
                            for r in readings):
            return f"premise {code} contradicted: {', '.join(f'{r.tag}={r.value:g}' for r in readings)} in range"
    elif code == "SILICA_HIGH":
        s = by_tag.get("silica_concentrate_pct")
        if s is not None and s.value < SILICA_HIGH_PCT:
            return f"premise SILICA_HIGH contradicted: silica_concentrate_pct={s.value:g} < {SILICA_HIGH_PCT}"
    elif code == "SENSOR_QUALITY":
        if all(s.quality == "good" for s in telemetry):
            return "premise SENSOR_QUALITY contradicted: all readings have good quality"
    return None


_SILICA_KINDS = {"lab_last", "lab_prev", "lab_streak", "soft_sensor", "silica_state"}
_REASON_FACT = {
    "PH_OUT_OF_RANGE": ("range", "ore_pulp_ph"),
    "DENSITY_OUT_OF_RANGE": ("range", "ore_pulp_density"),
}


def check_citations(action: ControlAction, ctx) -> tuple[str | None, str | None]:
    """(motivo de rechazo, aviso). Verifica que la evidencia citada existe y respalda la decisión.

    Solo se rechaza lo que puede ser *falso*: una cita a un hecho inexistente, o un motivo
    (SILICA_HIGH, PH_OUT_OF_RANGE, ...) sin un hecho citado que efectivamente lo muestre.
    Eso es una justificación inventada y bloquea un ADJUST.

    No citar el valor actual del tag que se mueve queda como aviso: el paso ya está acotado
    respecto de ese valor por la gramática (setpoint entero en [actual ± max_step]) y por
    `check_physics`, así que la omisión no hace inseguro el ajuste. Se intentó forzarlo con
    `prefixItems` en el schema y Ollama 0.12.6 no lo respeta.
    """
    if ctx is None:
        return None, None
    cited = []
    for fid in action.evidence:
        fact = ctx.get(fid)
        if fact is None:
            return f"cited fact {fid} does not exist", None
        cited.append(fact)
    false_claims, omissions = [], []
    if not cited:
        omissions.append("no evidence cited")
    if action.action == ActionType.ADJUST_SETPOINT.value:
        if not any(f.kind == "setpoint" and f.tag == action.target_tag for f in cited):
            omissions.append(f"ADJUST on {action.target_tag} without citing its current value")
    if action.reason_code == "SILICA_HIGH":
        if not any(f.kind in _SILICA_KINDS and f.flag in {"high", "sustained_high"} for f in cited):
            false_claims.append("SILICA_HIGH without citing a silica fact that is high")
    if action.reason_code in _REASON_FACT:
        kind, tag = _REASON_FACT[action.reason_code]
        if not any(f.kind == kind and f.tag == tag and f.flag == "out_of_range" for f in cited):
            false_claims.append(f"{action.reason_code} without citing {tag} out of range")
    warn = "citation check: " + "; ".join(false_claims + omissions) if (false_claims or omissions) else None
    if false_claims and action.action == ActionType.ADJUST_SETPOINT.value:
        return "citation check: " + "; ".join(false_claims), warn
    return None, warn


def failsafe_hold(reason: str) -> ControlAction:
    return ControlAction(
        action=ActionType.HOLD,
        reason_code=ReasonCode.OTHER,
        confidence=Confidence.LOW,
        rationale=f"failsafe: {reason}"[:200],
    )


def validate_output(
    raw: str, telemetry: list[TelemetrySample]
) -> tuple[ControlAction, str | None]:
    """Devuelve (acción ejecutable, motivo de rechazo o None)."""
    try:
        action = parse_action(raw)
        check_physics(action, telemetry)
        return action, None
    except OutputRejected as exc:
        return failsafe_hold(str(exc)), str(exc)
