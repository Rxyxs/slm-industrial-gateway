"""RouterAgent: guardrail de entrada + clasificación de intención (Agente 1 del pipeline).

Filtra patrones conocidos de ataque (inyección de prompt, SQL, comandos) de
forma determinista y sin costo de red antes de considerar invocar al SLM;
sanea el texto de entrada; y clasifica la intención de la consulta
(¿necesita herramientas analíticas o es una pregunta directa?) con una
llamada barata y de baja latencia al modelo (pocos tokens, temperatura 0).

Falla cerrado: cualquier entrada no reconocible como texto, amenaza detectada,
o salida del modelo ambigua se resuelve como `RequestIntent.REJECTED`, nunca
se asume la opción más permisiva.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from src.engine.llm_server import GenerationConfig, LLMServer

MAX_INPUT_LENGTH = 4000

# Caracteres de control C0 y DEL, salvo tab (\x09) y salto de línea (\x0a):
# se eliminan los bytes de control en sí (p. ej. el ESC de una secuencia
# ANSI), pero se conserva cualquier carácter imprimible que los acompañe.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

_DEFAULT_CLASSIFICATION_CONFIG = GenerationConfig(max_tokens=10, temperature=0.0)

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignor[ae]\w*\s+(las|all|previous|todas?)\s*(instrucciones|instructions)", re.IGNORECASE),
    re.compile(r"ignore\s+all\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"olvida\w*\s+todo\s+lo\s+anterior", re.IGNORECASE),
    re.compile(r"\bDAN\b"),
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"modo\s+desarrollador", re.IGNORECASE),
    re.compile(r"sin\s+restricciones", re.IGNORECASE),
    re.compile(r"unrestricted\s+AI", re.IGNORECASE),
    re.compile(r"bypass\s+(your\s+)?guardrails", re.IGNORECASE),
    re.compile(r"revela\s+tu\s+system\s+prompt", re.IGNORECASE),
]

_SQL_INJECTION_PATTERNS = [
    re.compile(r"'\s*OR\s*'1'\s*=\s*'1", re.IGNORECASE),
    re.compile(r";\s*(DROP|DELETE|UPDATE|INSERT)\s+", re.IGNORECASE),
    re.compile(r"\bUNION\s+SELECT\b", re.IGNORECASE),
]

_COMMAND_INJECTION_PATTERNS = [
    re.compile(r"\$\([^)]+\)"),
    re.compile(r"`[^`]+`"),
    re.compile(r"&&\s*rm\s+-rf"),
    re.compile(r"\.\./\.\./"),
]


class RequestIntent(str, Enum):
    """Resultado de clasificar una consulta de usuario."""

    ANALYTICS_REQUIRED = "ANALYTICS_REQUIRED"
    DIRECT_QA = "DIRECT_QA"
    REJECTED = "REJECTED"


def sanitize_input(text: str) -> str:
    """Limpia texto de entrada: quita caracteres de control, recorta espacios y trunca."""
    if not isinstance(text, str):
        raise ValueError("El texto de entrada debe ser una cadena de texto.")

    without_control = _CONTROL_CHARS_RE.sub("", text)
    return without_control.strip()[:MAX_INPUT_LENGTH]


def detect_threat(text: str) -> Optional[str]:
    """Detecta patrones conocidos de ataque sin invocar al modelo.

    Devuelve `"prompt_injection"`, `"sql_injection"`, `"command_injection"` o
    `None` si no se reconoce ninguno. Es una defensa determinista y barata,
    no un reemplazo de una clasificación semántica completa.
    """
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return "prompt_injection"
    for pattern in _SQL_INJECTION_PATTERNS:
        if pattern.search(text):
            return "sql_injection"
    for pattern in _COMMAND_INJECTION_PATTERNS:
        if pattern.search(text):
            return "command_injection"
    return None


class RouterAgent:
    """Guardrail de entrada y clasificador de intención de la consulta del usuario."""

    def __init__(
        self,
        llm_server: LLMServer,
        classification_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.llm_server = llm_server
        self.classification_config = classification_config or _DEFAULT_CLASSIFICATION_CONFIG

    def check_threat(self, text: str) -> Optional[str]:
        """Atajo sin costo de red al detector determinista de amenazas (`detect_threat`)."""
        return detect_threat(text)

    def classify_intent(self, query: Any) -> RequestIntent:
        """Clasifica `query`. Falla cerrado ante amenazas, entradas vacías o ambigüedad."""
        if not isinstance(query, str) or not query.strip():
            return RequestIntent.REJECTED

        if detect_threat(query) is not None:
            return RequestIntent.REJECTED

        sanitized = sanitize_input(query)
        if not sanitized:
            return RequestIntent.REJECTED

        prompt = (
            "Clasifica la siguiente consulta como ANALYTICS_REQUIRED (necesita "
            "herramientas analiticas: consultas SQL, deteccion de anomalias o "
            "calculo de RUL) o DIRECT_QA (pregunta general, se responde sin "
            f"herramientas). Responde solo con esa palabra.\n\nConsulta: {sanitized}"
        )
        raw = self.llm_server.generate(prompt, config=self.classification_config)
        normalized = raw.strip().upper().rstrip(".")

        if normalized == RequestIntent.ANALYTICS_REQUIRED.value:
            return RequestIntent.ANALYTICS_REQUIRED
        if normalized == RequestIntent.DIRECT_QA.value:
            return RequestIntent.DIRECT_QA
        return RequestIntent.REJECTED
