"""Contratos de entrada/salida del gateway.

Hay dos contratos de salida:

- `ModelOutput` es lo que el modelo puede producir. Su JSON Schema se envía al
  motor como restricción de decodificación (Ollama `format`, vLLM
  `response_format`). Es una unión discriminada por `action`: la variante HOLD
  no tiene campos de setpoint, así que la gramática hace imposible un "HOLD con
  setpoint". Medido en Ollama 0.12.6: la gramática respeta enums, constantes,
  `minimum`/`maximum` de enteros y `maxLength` de strings, pero NO `minimum`/`maximum`
  de números decimales (con `confidence: float` en [0, 1] el modelo devolvía 3). Por
  eso la confianza es categórica y, cuando hay lecturas actuales, el setpoint se pide
  como entero acotado a [actual - max_step, actual + max_step] dentro del rango del
  tag: un valor inseguro no se puede ni escribir. La validación física de
  `validation.py` se mantiene igual, para la nube y como segunda capa.
- `ControlAction` es la vista plana y estable que recibe el SCADA/PLC.

Igual se vuelve a validar todo al recibirlo: un backend en la nube puede ignorar
la restricción.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from src.gateway.tags import TAG_REGISTRY, WRITABLE_TAGS

WritableTag = Enum("WritableTag", {t.upper(): t for t in WRITABLE_TAGS}, type=str)


class ActionType(str, Enum):
    HOLD = "HOLD"
    ADJUST_SETPOINT = "ADJUST_SETPOINT"
    RAISE_ALARM = "RAISE_ALARM"


class ReasonCode(str, Enum):
    WITHIN_NORMAL = "WITHIN_NORMAL"
    SILICA_HIGH = "SILICA_HIGH"
    PH_OUT_OF_RANGE = "PH_OUT_OF_RANGE"
    DENSITY_OUT_OF_RANGE = "DENSITY_OUT_OF_RANGE"
    LEVEL_DEVIATION = "LEVEL_DEVIATION"
    AIR_FLOW_DEVIATION = "AIR_FLOW_DEVIATION"
    SENSOR_QUALITY = "SENSOR_QUALITY"
    OTHER = "OTHER"


class Confidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TelemetrySample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag: str = Field(pattern=r"^[a-z0-9_]{1,64}$")
    value: float
    quality: Literal["good", "uncertain", "bad"] = "good"
    ts: Optional[datetime] = None

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value must be finite")
        return v

    @field_validator("tag")
    @classmethod
    def _known_tag(cls, v: str) -> str:
        if v not in TAG_REGISTRY:
            raise ValueError(f"unknown tag '{v}'")
        return v


class HourIn(BaseModel):
    """Promedio horario de tags, como lo entrega un historian."""

    model_config = ConfigDict(extra="forbid")
    ts: datetime
    values: dict[str, float]

    @field_validator("values")
    @classmethod
    def _known(cls, v: dict[str, float]) -> dict[str, float]:
        bad = [k for k in v if k not in TAG_REGISTRY]
        if bad:
            raise ValueError(f"unknown tags {bad}")
        if not all(math.isfinite(x) for x in v.values()):
            raise ValueError("values must be finite")
        return v


class LabIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    sampled_at: datetime
    silica_pct: float = Field(ge=0, le=100)
    delay_h: int = Field(default=2, ge=0, le=48)


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    telemetry: list[TelemetrySample] = Field(min_length=1, max_length=64)
    # Nota libre del operador. Es entrada no confiable: pasa por el inspector de
    # prompt injection y llega al modelo delimitada como dato, nunca como orden.
    operator_note: str = Field(default="", max_length=500)
    mode: Literal["advisory", "closed_loop"] = "advisory"
    # Contexto opcional. Si no viene, el gateway usa lo que acumuló en su ContextStore.
    history: Optional[list[HourIn]] = Field(default=None, max_length=48)
    lab_results: Optional[list[LabIn]] = Field(default=None, max_length=96)


def _rationale():
    return Field(description="One short sentence, under 25 words")


def _evidence():
    return Field(default_factory=list, max_length=6,
                 description="Ids of the facts this decision rests on, e.g. [\"F1\", \"F2\"]")


class HoldOutput(BaseModel):
    """Dejar todo como está."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["HOLD"]
    reason_code: ReasonCode
    confidence: Confidence
    rationale: str = _rationale()
    evidence: list[str] = _evidence()


class AdjustOutput(BaseModel):
    """Mover un único setpoint escribible."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["ADJUST_SETPOINT"]
    target_tag: WritableTag
    new_value: float = Field(description="New setpoint in the tag's unit, inside its range and "
                             "within max_step of the current reading")
    reason_code: ReasonCode
    confidence: Confidence
    rationale: str = _rationale()
    evidence: list[str] = _evidence()


class AlarmOutput(BaseModel):
    """Requiere decisión humana."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["RAISE_ALARM"]
    reason_code: ReasonCode
    confidence: Confidence
    rationale: str = _rationale()
    evidence: list[str] = _evidence()


ModelOutput = Annotated[Union[HoldOutput, AdjustOutput, AlarmOutput], Field(discriminator="action")]
MODEL_OUTPUT = TypeAdapter(ModelOutput)


class ControlAction(BaseModel):
    """Acción plana entregada al SCADA. `extra="forbid"`: un campo inventado invalida todo."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    action: ActionType
    target_tag: Optional[WritableTag] = None
    new_value: Optional[float] = None
    reason_code: ReasonCode
    confidence: Confidence
    # El texto explicativo no es ejecutable; se recorta en vez de rechazar la acción.
    rationale: str = Field(max_length=200)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "ControlAction":
        if self.action == ActionType.ADJUST_SETPOINT.value:
            if self.target_tag is None or self.new_value is None:
                raise ValueError("ADJUST_SETPOINT requires target_tag and new_value")
            if not math.isfinite(self.new_value):
                raise ValueError("new_value must be finite")
        elif self.target_tag is not None or self.new_value is not None:
            raise ValueError(f"{self.action} must not carry target_tag/new_value")
        return self

    @classmethod
    def from_output(cls, out) -> "ControlAction":
        data = out.model_dump(mode="json")
        data["rationale"] = " ".join(data["rationale"].split())[:200]
        return cls.model_validate(data)


class InferenceResponse(BaseModel):
    request_id: str
    asset_id: str
    action: ControlAction
    backend: str
    tier: Literal["edge", "cloud", "failsafe", "cache"]
    latency_ms: float
    cached: bool = False
    degraded: bool = False
    rejected_reason: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    attempts: list[str] = Field(default_factory=list)


class JobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    result: Optional[InferenceResponse] = None
    error: Optional[str] = None


_FIELD_ORDER = ("evidence", "reason_code", "action", "target_tag", "new_value", "confidence", "rationale")


def setpoint_bounds(telemetry) -> dict[str, tuple[int, int]]:
    """Rango entero que la gramática le permite a cada tag escribible en esta solicitud."""
    out = {}
    for s in telemetry:
        spec = TAG_REGISTRY.get(s.tag)
        if spec is None or not spec.writable or s.quality != "good":
            continue
        lo = math.ceil(max(spec.low, s.value - spec.max_step))
        hi = math.floor(min(spec.high, s.value + spec.max_step))
        if lo <= hi:
            out[s.tag] = (lo, hi)
    return out


def model_output_json_schema(fact_ids: list[str] | None = None,
                             bounds: dict[str, tuple[int, int]] | None = None) -> dict:
    """JSON Schema autocontenido para guided decoding.

    Con `fact_ids` (modo contexto), `evidence` pasa a ser obligatorio y sus elementos un
    enum de los hechos de *esta* solicitud: la gramática no deja citar un hecho que no
    existe. Además se reordenan las propiedades para que el modelo escriba primero la
    evidencia y el motivo, y recién después la acción: la gramática genera en el orden del
    schema, así que el orden es la cadena de razonamiento.
    """
    schema = copy.deepcopy(MODEL_OUTPUT.json_schema())
    defs = schema["$defs"]
    for name in ("HoldOutput", "AdjustOutput", "AlarmOutput"):
        props = defs[name]["properties"]
        props["rationale"]["maxLength"] = 160
        if fact_ids:
            props["evidence"] = {"type": "array", "items": {"enum": list(fact_ids)}, "minItems": 1,
                                 "maxItems": 6, "description": props["evidence"]["description"]}
        else:
            props.pop("evidence", None)
        defs[name]["properties"] = {k: props[k] for k in _FIELD_ORDER if k in props}
        defs[name]["required"] = [k for k in _FIELD_ORDER if k in defs[name]["properties"]]
    if bounds is None:
        return schema
    # Una variante ADJUST por tag escribible con lectura válida, con su rango entero.
    adjust = defs.pop("AdjustOutput")
    variants = [{"$ref": "#/$defs/HoldOutput"}, {"$ref": "#/$defs/AlarmOutput"}]
    for tag, (lo, hi) in bounds.items():
        v = copy.deepcopy(adjust)
        v["properties"]["target_tag"] = {"const": tag}
        v["properties"]["new_value"] = {"type": "integer", "minimum": lo, "maximum": hi,
                                        "description": f"new {tag} setpoint, between {lo} and {hi}"}
        name = f"Adjust_{tag}"
        defs[name] = v
        variants.append({"$ref": f"#/$defs/{name}"})
    schema.pop("discriminator", None)
    schema.pop("oneOf", None)
    schema["anyOf"] = variants
    return schema


def control_action_json_schema() -> dict:
    """JSON Schema de la acción que consume el SCADA."""
    return ControlAction.model_json_schema()
