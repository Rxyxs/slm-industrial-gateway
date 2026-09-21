"""VerifierAgent: última puerta de calidad del pipeline, antes de responder al cliente.

Dos chequeos, ambos locales y sin dependencia de red — a diferencia de
`src.evaluation.FaithfulnessEvaluator`, que usa un LLM juez externo y está
pensado para evaluación *offline/batch* (ver README), no para el camino
caliente de inferencia:

1. **Formato**: la respuesta final debe ser texto no vacío y no puede ser un
   tool-call sin resolver (`{"tool": ..., "arguments": {...}}`) — si el SLM
   queda "atascado" pidiendo otra tool en vez de redactar una respuesta, se
   rechaza en vez de filtrar JSON crudo al cliente.
2. **Fidelidad**: si se ejecutó una tool, todo valor numérico mencionado en la
   respuesta debe poder rastrearse hasta el resultado crudo de esa tool. Es
   una verificación heurística y barata (comparación de números, no un LLM
   juez), pensada como red de seguridad en tiempo real, no como reemplazo del
   `FaithfulnessMetric` de DeepEval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Tuple

from src.engine import GenerationError
from src.guardrails.validators import GuardrailError, OutputValidationError, validate_json_output

from .schemas import ToolCallEnvelope

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


class FaithfulnessError(GuardrailError):
    """Se lanza cuando la respuesta menciona valores no sustentados por el contexto de las tools."""


@dataclass
class VerificationResult:
    """Resultado de una verificación exitosa (si falla, `verify()` lanza una excepción)."""

    is_faithful: bool
    is_valid_format: bool
    checked_claims: List[str] = field(default_factory=list)


def _extract_numbers(text: str) -> Set[str]:
    return {match.replace(",", ".") for match in _NUMBER_RE.findall(text)}


class VerifierAgent:
    """Verifica la respuesta final del SLM antes de que la API la devuelva al cliente."""

    def _is_unresolved_tool_call(self, response_text: str) -> bool:
        try:
            validate_json_output(response_text, ToolCallEnvelope)
        except OutputValidationError:
            return False
        return True

    def _check_faithfulness(self, response_text: str, tool_context: Any) -> Tuple[bool, List[str]]:
        if tool_context is None:
            return True, []

        response_numbers = _extract_numbers(response_text)
        context_numbers = _extract_numbers(json.dumps(tool_context, ensure_ascii=False, default=str))
        unsupported = sorted(response_numbers - context_numbers)
        return not unsupported, unsupported

    def verify(self, response_text: str, tool_context: Optional[Any] = None) -> VerificationResult:
        """Verifica `response_text`.

        Lanza `GenerationError` si está vacía/no es texto (falla del modelo, no
        de contenido), `OutputValidationError` si es un tool-call sin resolver,
        o `FaithfulnessError` si contiene valores no sustentados por
        `tool_context`. Devuelve un `VerificationResult` si pasa todo.
        """
        if not isinstance(response_text, str) or not response_text.strip():
            raise GenerationError("El modelo produjo una respuesta vacía o inválida.")

        if self._is_unresolved_tool_call(response_text):
            raise OutputValidationError(
                "La respuesta final es un tool-call sin resolver, no una respuesta en lenguaje natural."
            )

        is_faithful, unsupported = self._check_faithfulness(response_text, tool_context)
        if not is_faithful:
            raise FaithfulnessError(
                "La respuesta menciona valores que no aparecen en el contexto de las herramientas "
                f"ejecutadas: {unsupported}"
            )

        return VerificationResult(is_faithful=True, is_valid_format=True, checked_claims=unsupported)
