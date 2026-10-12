"""Contratos de entrada/salida del gateway.

Hay dos contratos de salida:

- `ModelOutput` es lo que el modelo puede producir. Su JSON Schema se envía al
  motor como restricción de decodificación (Ollama `format`, vLLM
  `response_format`). Es una unión discriminada por `action`: la variante HOLD
  no tiene campos de setpoint, así que la gramática hace imposible un "HOLD con
  setpoint". La confianza es categórica por la misma razón: la gramática de
  llama.cpp/Ollama respeta enums y constantes, pero ignora `minimum`/`maximum`
  y `maxLength` (medido: con `confidence: float` en [0, 1] el modelo devolvía 3).
  Se usan solo restricciones que el motor hace cumplir; los números (el
  setpoint) se validan después contra la física, en `validation.py`.
- `ControlAction` es la vista plana y estable que recibe el SCADA/PLC.

Igual se vuelve a validar todo al recibirlo: un backend en la nube puede ignorar
la restricción.
"""

from __future__ import annotations

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


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    telemetry: list[TelemetrySample] = Field(min_length=1, max_length=64)
    # Nota libre del operador. Es entrada no confiable: pasa por el inspector de
    # prompt injection y llega al modelo delimitada como dato, nunca como orden.
    operator_note: str = Field(default="", max_length=500)
    mode: Literal["advisory", "closed_loop"] = "advisory"


def _rationale():
    return Field(description="One short sentence, under 25 words")


class HoldOutput(BaseModel):
    """Dejar todo como está."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["HOLD"]
    reason_code: ReasonCode
    confidence: Confidence
    rationale: str = _rationale()


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


class AlarmOutput(BaseModel):
    """Requiere decisión humana."""

    model_config = ConfigDict(extra="forbid")
    action: Literal["RAISE_ALARM"]
    reason_code: ReasonCode
    confidence: Confidence
    rationale: str = _rationale()


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


def model_output_json_schema() -> dict:
    """JSON Schema autocontenido para guided decoding."""
    return MODEL_OUTPUT.json_schema()


def control_action_json_schema() -> dict:
    """JSON Schema de la acción que consume el SCADA."""
    return ControlAction.model_json_schema()
