"""Registro central de herramientas, con esquemas compatibles con OpenAI Function Calling."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Type, Union

from pydantic import BaseModel, ValidationError

from src.guardrails.validators import GuardrailError, OutputValidationError, validate_json_output


class ToolNotFoundError(KeyError):
    """Se lanza al solicitar una herramienta que no está registrada."""


class ToolExecutionError(RuntimeError):
    """Se lanza cuando una herramienta recibe argumentos inválidos o falla al ejecutarse."""


class RegisteredTool:
    """Une el nombre, la descripción, el esquema de entrada y la función de una herramienta."""

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        func: Callable[[BaseModel], Any],
    ) -> None:
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func

    def to_openai_schema(self) -> Dict[str, Any]:
        """Genera el esquema JSON de la herramienta en formato OpenAI Function Calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }

    def run(self, arguments: Union[Dict[str, Any], str]) -> Any:
        """Valida `arguments` (dict o cadena JSON) contra `args_schema` y ejecuta la herramienta."""
        if isinstance(arguments, (str, bytes)):
            try:
                validated = validate_json_output(arguments, self.args_schema)
            except OutputValidationError as exc:
                raise ToolExecutionError(str(exc)) from exc
        else:
            try:
                validated = self.args_schema.model_validate(arguments)
            except ValidationError as exc:
                raise ToolExecutionError(f"Argumentos inválidos para '{self.name}': {exc}") from exc

        try:
            return self.func(validated)
        except GuardrailError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Error al ejecutar '{self.name}': {exc}") from exc


class ToolRegistry:
    """Registro central de herramientas disponibles para el agente."""

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        func: Callable[[BaseModel], Any],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"La herramienta '{name}' ya está registrada.")
        self._tools[name] = RegisteredTool(name, description, args_schema, func)

    def register(
        self, name: str, description: str, args_schema: Type[BaseModel]
    ) -> Callable[[Callable[[BaseModel], Any]], Callable[[BaseModel], Any]]:
        """Decorador de conveniencia para registrar una función como herramienta."""

        def decorator(func: Callable[[BaseModel], Any]) -> Callable[[BaseModel], Any]:
            self.register_tool(name, description, args_schema, func)
            return func

        return decorator

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Herramienta no registrada: '{name}'") from exc

    def list_tools(self) -> List[str]:
        return sorted(self._tools)

    def to_openai_schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def dispatch(self, name: str, arguments: Union[Dict[str, Any], str]) -> Any:
        """Despacha una llamada a función por nombre, tal como la emitiría un LLM."""
        return self.get(name).run(arguments)
