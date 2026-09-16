"""Motor de inferencia local para modelos SLM (Qwen-2.5 / Llama-3.1) en GGUF."""

from .benchmarks import BenchmarkResult, current_ram_mb, current_vram_mb, run_benchmark
from .llm_server import (
    GenerationConfig,
    GenerationError,
    LLMServer,
    ModelLoadError,
    detect_gpu_layers,
)

__all__ = [
    "LLMServer",
    "GenerationConfig",
    "ModelLoadError",
    "GenerationError",
    "detect_gpu_layers",
    "BenchmarkResult",
    "run_benchmark",
    "current_ram_mb",
    "current_vram_mb",
]
