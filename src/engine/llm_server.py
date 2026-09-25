"""Servidor de inferencia local para modelos SLM en formato GGUF.

Usa llama-cpp-python como backend y soporta fallback dinámico de GPU a CPU
cuando la carga en GPU falla (VRAM insuficiente, build sin soporte CUDA, etc.).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - en tests se sustituye vía mock
    Llama = None  # type: ignore[assignment]

from src.telemetry import trace_stage
from src.telemetry.logger import STAGE_MODEL_INFERENCE, STAGE_TOKENIZER

logger = logging.getLogger(__name__)

# agent_name de las etapas instrumentadas acá: no son un agente del pipeline
# (src/agents/), son el motor de inferencia en sí -- lo que `AgentOrchestrator`
# invoca como una caja negra desde `_propose()`.
ENGINE_AGENT_NAME = "engine"

DEFAULT_CONTEXT_WINDOW = 4096
DEFAULT_MAX_TOKENS = 512


class ModelLoadError(RuntimeError):
    """Error al cargar el modelo GGUF (archivo inválido, backend ausente, etc.)."""


class GenerationError(RuntimeError):
    """Error durante la generación de texto (síncrona o por streaming)."""


def detect_gpu_layers(requested: Optional[int] = None) -> int:
    """Resuelve cuántas capas delegar a GPU.

    Si el usuario especifica un valor, se respeta tal cual. En caso contrario
    se detecta la presencia de una GPU NVIDIA vía `nvidia-smi`: -1 (todas las
    capas a GPU) si se encuentra una, 0 (CPU) si no.
    """
    if requested is not None:
        return requested

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return 0

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0

    if result.returncode == 0 and result.stdout.strip():
        return -1
    return 0


@dataclass
class GenerationConfig:
    """Parámetros de muestreo para una generación."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)


class LLMServer:
    """Servidor de inferencia local para modelos GGUF (Qwen-2.5 / Llama-3.1)."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = DEFAULT_CONTEXT_WINDOW,
        n_gpu_layers: Optional[int] = None,
        n_threads: Optional[int] = None,
        verbose: bool = False,
        **model_kwargs: Any,
    ) -> None:
        if Llama is None:
            raise ModelLoadError(
                "llama-cpp-python no está instalado. Instálalo con "
                "`pip install llama-cpp-python` para usar LLMServer."
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelLoadError(f"No se encontró el modelo en: {self.model_path}")

        self.n_ctx = n_ctx
        self.n_gpu_layers = detect_gpu_layers(n_gpu_layers)

        try:
            self._llm = self._load(self.n_gpu_layers, n_threads, verbose, model_kwargs)
        except Exception as primary_exc:  # noqa: BLE001 - se reintenta en CPU
            if self.n_gpu_layers == 0:
                raise ModelLoadError(f"Error al cargar el modelo: {primary_exc}") from primary_exc

            logger.warning(
                "Fallo al cargar el modelo en GPU (%s); reintentando en CPU.", primary_exc
            )
            self.n_gpu_layers = 0
            try:
                self._llm = self._load(0, n_threads, verbose, model_kwargs)
            except Exception as fallback_exc:  # noqa: BLE001
                raise ModelLoadError(
                    f"Error al cargar el modelo tanto en GPU como en CPU: {fallback_exc}"
                ) from fallback_exc

        logger.info(
            "Modelo cargado: %s (n_ctx=%d, n_gpu_layers=%d)",
            self.model_path.name,
            self.n_ctx,
            self.n_gpu_layers,
        )

    def _load(
        self,
        n_gpu_layers: int,
        n_threads: Optional[int],
        verbose: bool,
        model_kwargs: dict[str, Any],
    ) -> Any:
        return Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
            **model_kwargs,
        )

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Genera texto de forma síncrona y devuelve la respuesta completa.

        Dos etapas medidas por separado (`src.telemetry.trace_stage`):
        tokenizar el prompt de entrada (`STAGE_TOKENIZER`) y la llamada real
        al motor de inferencia (`STAGE_MODEL_INFERENCE`) -- para poder ver,
        en el log estructurado, cuál de las dos domina la latencia de una
        solicitud lenta en vez de un único número agregado.
        """
        if not prompt:
            raise ValueError("El prompt no puede estar vacío.")

        cfg = config or GenerationConfig()

        with trace_stage(STAGE_TOKENIZER, ENGINE_AGENT_NAME) as span:
            prompt_tokens = self._llm.tokenize(prompt.encode("utf-8"))
            span.token_count = len(prompt_tokens)

        with trace_stage(STAGE_MODEL_INFERENCE, ENGINE_AGENT_NAME) as span:
            try:
                output = self._llm(
                    prompt,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    repeat_penalty=cfg.repeat_penalty,
                    stop=cfg.stop or None,
                    stream=False,
                )
            except Exception as exc:  # noqa: BLE001
                raise GenerationError(f"Error durante la generación: {exc}") from exc
            span.token_count = output.get("usage", {}).get("completion_tokens")

        try:
            return output["choices"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError(f"Respuesta del modelo con formato inesperado: {exc}") from exc

    def generate_stream(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """Genera texto por streaming, produciendo fragmentos a medida que llegan."""
        if not prompt:
            raise ValueError("El prompt no puede estar vacío.")

        cfg = config or GenerationConfig()
        try:
            stream = self._llm(
                prompt,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repeat_penalty=cfg.repeat_penalty,
                stop=cfg.stop or None,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Error durante la generación: {exc}") from exc

        try:
            for chunk in stream:
                try:
                    text = chunk["choices"][0]["text"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise GenerationError(
                        f"Fragmento de streaming con formato inesperado: {exc}"
                    ) from exc
                if text:
                    yield text
        except GenerationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Error durante el streaming: {exc}") from exc

    def close(self) -> None:
        """Libera el contexto de llama.cpp asociado al modelo."""
        llm = getattr(self, "_llm", None)
        if llm is not None and hasattr(llm, "close"):
            llm.close()

    def __enter__(self) -> "LLMServer":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()
