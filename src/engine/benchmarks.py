"""Medición de rendimiento del motor de inferencia: t/s, TTFT y memoria."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - opcional, degrada a None
    psutil = None  # type: ignore[assignment]

from .llm_server import GenerationConfig, LLMServer


@dataclass
class BenchmarkResult:
    """Resultado de una corrida de benchmark de generación."""

    ttft_seconds: float
    total_seconds: float
    tokens_generated: int
    tokens_per_second: float
    ram_mb: Optional[float]
    vram_mb: Optional[float]


def current_ram_mb() -> Optional[float]:
    """RAM residente (RSS) del proceso actual en MB, o None si psutil no está disponible."""
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def current_vram_mb() -> Optional[float]:
    """VRAM usada según `nvidia-smi`, o None si no hay GPU NVIDIA disponible."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        return float(result.stdout.strip().splitlines()[0])
    except ValueError:
        return None


def run_benchmark(
    server: LLMServer, prompt: str, config: Optional[GenerationConfig] = None
) -> BenchmarkResult:
    """Ejecuta una generación por streaming y mide TTFT, tokens/segundo y memoria."""
    start = time.perf_counter()
    first_token_time: Optional[float] = None
    # Cada fragmento de streaming de llama.cpp corresponde a ~1 token.
    tokens_generated = 0

    for _chunk in server.generate_stream(prompt, config=config):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        tokens_generated += 1

    end = time.perf_counter()

    ttft = (first_token_time or end) - start
    total = end - start
    tokens_per_second = tokens_generated / total if total > 0 else 0.0

    return BenchmarkResult(
        ttft_seconds=ttft,
        total_seconds=total,
        tokens_generated=tokens_generated,
        tokens_per_second=tokens_per_second,
        ram_mb=current_ram_mb(),
        vram_mb=current_vram_mb(),
    )
