"""Guardrails de seguridad: bloqueo de SQL peligroso y validación estricta de salidas JSON."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Type, Union

from pydantic import BaseModel, ValidationError


class GuardrailError(Exception):
    """Excepción base para cualquier violación de guardrail."""


class DangerousSQLError(GuardrailError):
    """Se lanza cuando una consulta SQL es potencialmente destructiva o no está permitida."""


class OutputValidationError(GuardrailError):
    """Se lanza cuando una salida no es JSON válido o no cumple el esquema esperado."""


# El agente solo debe poder leer datos: cualquier otra sentencia queda fuera por defecto.
ALLOWED_SQL_VERBS = frozenset({"SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW"})

# Defensa en profundidad: se bloquean aunque aparezcan dentro de subconsultas o CTEs.
BLOCKED_SQL_KEYWORDS = frozenset({
    "DROP", "DELETE", "ALTER", "TRUNCATE", "INSERT", "UPDATE", "CREATE",
    "GRANT", "REVOKE", "ATTACH", "DETACH", "EXEC", "EXECUTE", "COPY",
    "MERGE", "REPLACE", "CALL", "VACUUM", "LOAD", "PRAGMA",
})

_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LEADING_VERB_RE = re.compile(r"^\s*([A-Za-z]+)")


def _strip_sql_comments(query: str) -> str:
    without_block_comments = _BLOCK_COMMENT_RE.sub(" ", query)
    return _LINE_COMMENT_RE.sub(" ", without_block_comments)


def validate_sql_query(query: str) -> str:
    """Valida que `query` sea una única sentencia SQL de solo lectura sin palabras clave peligrosas.

    Devuelve la sentencia saneada (sin comentarios ni ';' final) si es segura,
    o lanza `DangerousSQLError` en caso contrario.
    """
    if not isinstance(query, str) or not query.strip():
        raise DangerousSQLError("La consulta SQL está vacía o no es una cadena de texto válida.")

    cleaned = _strip_sql_comments(query)
    statements = [statement.strip() for statement in cleaned.split(";") if statement.strip()]

    if not statements:
        raise DangerousSQLError("La consulta SQL está vacía tras eliminar comentarios.")
    if len(statements) > 1:
        raise DangerousSQLError(
            "Solo se permite una única sentencia SQL por consulta (se detectaron múltiples)."
        )

    statement = statements[0]
    match = _LEADING_VERB_RE.match(statement)
    verb = match.group(1).upper() if match else ""

    if verb not in ALLOWED_SQL_VERBS:
        raise DangerousSQLError(
            f"Sentencia SQL no permitida: '{verb or statement[:20]}'. "
            f"Solo se permiten consultas de lectura: {', '.join(sorted(ALLOWED_SQL_VERBS))}."
        )

    upper_statement = statement.upper()
    for keyword in BLOCKED_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_statement):
            raise DangerousSQLError(f"Consulta bloqueada: contiene la palabra clave peligrosa '{keyword}'.")

    return statement


def validate_json_output(raw_output: Union[str, bytes, Dict[str, Any]], schema: Type[BaseModel]) -> BaseModel:
    """Parsea `raw_output` como JSON y lo valida estrictamente contra `schema`.

    Lanza `OutputValidationError` si el JSON está malformado, si el nivel superior
    no es un objeto, o si no cumple el esquema (sin coerción de tipos entre str/num/bool).
    """
    if isinstance(raw_output, (str, bytes)):
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise OutputValidationError(f"JSON malformado: {exc}") from exc
    elif isinstance(raw_output, dict):
        data = raw_output
    else:
        raise OutputValidationError(
            f"La salida debe ser una cadena JSON o un dict, no {type(raw_output).__name__}."
        )

    if not isinstance(data, dict):
        raise OutputValidationError("El JSON de salida debe ser un objeto en el nivel superior.")

    try:
        return schema.model_validate(data, strict=True)
    except ValidationError as exc:
        raise OutputValidationError(f"El JSON no cumple el esquema '{schema.__name__}': {exc}") from exc
