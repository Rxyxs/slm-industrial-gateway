"""SafetyComplianceAgent: última puerta de seguridad física del pipeline,
después de `VerifierAgent` y antes de que la respuesta llegue al cliente.

Audita la recomendación final en lenguaje natural contra una matriz FIJA de
límites operativos de diseño (potencia MW, presión PSI, temperatura °C). Si
algún valor numérico mencionado en la respuesta -- con su unidad -- supera el
límite de diseño del sitio, intercepta el flujo (nunca deja pasar la
respuesta) y lanza `SafetyAlertError` con código `SAFETY_ALERT`.

Igual que `VerifierAgent` (ver su docstring): un chequeo local, determinista y
sin dependencia de red -- regex + comparación numérica, no un LLM juez -- para
que audite en el camino caliente de inferencia sin agregar latencia real.

La matriz de límites es una constante de módulo, envuelta en
`types.MappingProxyType` (inmutable en runtime: escribirle lanza `TypeError`)
y compuesta por dataclasses `frozen=True`. No se lee de variables de entorno,
de configuración externa, ni de nada que un prompt o una tool pueda influir
-- son límites de diseño físico del sitio, no una preferencia ajustable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import List, Mapping, Tuple

from src.guardrails.validators import GuardrailError


@dataclass(frozen=True)
class OperationalLimit:
    """Límite de diseño físico fijo para una magnitud del sitio."""

    quantity: str
    unit: str
    max_value: float
    description: str


# --------------------------------------------------------------------------- #
# Matriz de límites operativos -- fija e inmutable, codificada acá a propósito.
# --------------------------------------------------------------------------- #

OPERATIONAL_LIMITS: Mapping[str, OperationalLimit] = MappingProxyType({
    "power_mw": OperationalLimit(
        quantity="potencia",
        unit="MW",
        max_value=150.0,
        description="Potencia máxima de diseño del generador/motor principal del sitio.",
    ),
    "pressure_psi": OperationalLimit(
        quantity="presión",
        unit="PSI",
        max_value=3000.0,
        description="Presión máxima de diseño del circuito hidráulico/vasija del sitio.",
    ),
    "temperature_c": OperationalLimit(
        quantity="temperatura",
        unit="°C",
        max_value=650.0,
        description="Temperatura crítica máxima de operación segura del equipo del sitio.",
    ),
})

# Un patrón por límite, atado a su clave en `OPERATIONAL_LIMITS` -- agregar un
# límite nuevo implica agregar acá su patrón, nunca al revés (el patrón nunca
# decide el límite, solo qué texto pertenece a qué magnitud).
_PATTERN_BY_LIMIT: Mapping[str, "re.Pattern[str]"] = MappingProxyType({
    "power_mw": re.compile(r"(-?\d+(?:[.,]\d+)?)\s*MW\b", re.IGNORECASE),
    "pressure_psi": re.compile(r"(-?\d+(?:[.,]\d+)?)\s*PSI\b", re.IGNORECASE),
    "temperature_c": re.compile(r"(-?\d+(?:[.,]\d+)?)\s*(?:°\s*C\b|grados\s*C(?:elsius)?\b)", re.IGNORECASE),
})


@dataclass(frozen=True)
class SafetyViolation:
    """Un valor puntual de la respuesta que supera su límite de diseño."""

    quantity: str
    value: float
    limit: float
    unit: str
    description: str


class SafetyAlertError(GuardrailError):
    """SAFETY_ALERT -- se lanza cuando la respuesta propone uno o más valores
    que superan los límites de diseño fijados en `OPERATIONAL_LIMITS`. El
    flujo se interrumpe acá: quien llame a `audit` nunca recibe una
    recomendación fuera de rango, ni parcial ni completa.
    """

    code = "SAFETY_ALERT"

    def __init__(self, message: str, violations: Tuple[SafetyViolation, ...]) -> None:
        super().__init__(message)
        self.violations = violations


@dataclass
class SafetyCheckResult:
    """Resultado de una auditoría exitosa (si falla, `audit` lanza `SafetyAlertError`)."""

    is_safe: bool
    checked_values: List[Tuple[str, float, str]] = field(default_factory=list)


def _extract_checked_and_violations(
    response_text: str,
) -> Tuple[List[Tuple[str, float, str]], List[SafetyViolation]]:
    checked: List[Tuple[str, float, str]] = []
    violations: List[SafetyViolation] = []

    for limit_key, pattern in _PATTERN_BY_LIMIT.items():
        limit = OPERATIONAL_LIMITS[limit_key]
        for raw_value in pattern.findall(response_text):
            value = float(raw_value.replace(",", "."))
            checked.append((limit.quantity, value, limit.unit))
            if value > limit.max_value:
                violations.append(
                    SafetyViolation(
                        quantity=limit.quantity,
                        value=value,
                        limit=limit.max_value,
                        unit=limit.unit,
                        description=limit.description,
                    )
                )

    return checked, violations


class SafetyComplianceAgent:
    """Audita que la recomendación final no proponga valores fuera de los
    límites de diseño físico del sitio, antes de que la respuesta se entregue.
    """

    def audit(self, response_text: str) -> SafetyCheckResult:
        """Audita `response_text`.

        Lanza `SafetyAlertError` (código `SAFETY_ALERT`) si algún valor
        numérico con unidad de potencia/presión/temperatura supera su límite
        de diseño. Devuelve `SafetyCheckResult(is_safe=True, ...)` si todos
        los valores encontrados -- puede que ninguno -- están dentro de rango.
        """
        if not isinstance(response_text, str):
            raise TypeError("SafetyComplianceAgent.audit requiere una respuesta de tipo str.")

        checked, violations = _extract_checked_and_violations(response_text)

        if violations:
            detail = "; ".join(
                f"{v.quantity} propuesta de {v.value} {v.unit} supera el límite de diseño de "
                f"{v.limit} {v.unit} ({v.description})"
                for v in violations
            )
            raise SafetyAlertError(f"SAFETY_ALERT: {detail}", violations=tuple(violations))

        return SafetyCheckResult(is_safe=True, checked_values=checked)
