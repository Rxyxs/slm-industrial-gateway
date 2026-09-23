"""Medición de rendimiento del motor de inferencia: t/s, TTFT y memoria."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - opcional, degrada a None
    psutil = None  # type: ignore[assignment]

from .llm_server import DEFAULT_CONTEXT_WINDOW, GenerationConfig, LLMServer


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


DEFAULT_RESULT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "reports" / "benchmark_result.json"

DEFAULT_BENCHMARK_PROMPT = (
    "El sensor de vibración del rodamiento del molino SAG 3 muestra una tendencia "
    "creciente en las últimas 48 horas. Explica en un párrafo qué pasos de "
    "mantenimiento predictivo recomendarías antes de programar una detención."
)


def _run_cli() -> None:  # pragma: no cover - I/O real, cubierto por inspección manual
    """CLI para medir TTFT/tokens-por-segundo/RAM de un modelo GGUF real.

    Fuerza `n_gpu_layers=0` (CPU) explícitamente en vez de dejar que
    `LLMServer` autodetecte GPU (`detect_gpu_layers`): este comando existe
    específicamente para reportar un número de CPU/edge real y etiquetado
    como tal, y autodetectar GPU aquí produciría, en una máquina con GPU
    disponible, un número que no es el que el nombre del comando promete.
    """
    import argparse
    import json as json_module

    parser = argparse.ArgumentParser(description="Benchmark real de inferencia (TTFT, tokens/s, RAM) sobre CPU.")
    parser.add_argument("--model-path", required=True, help="Ruta al archivo .gguf a benchmarkear.")
    parser.add_argument(
        "--model-label", default=None,
        help="Nombre legible del modelo para el resultado guardado (default: nombre del archivo). "
             "El archivo .gguf en sí no lleva metadata de qué checkpoint es una vez renombrado a "
             "model.gguf, así que el label es la única forma de que el reporte diga qué se midió.",
    )
    parser.add_argument("--prompt", default=DEFAULT_BENCHMARK_PROMPT, help="Prompt de la corrida de benchmark.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Tokens máximos a generar (default: 128).")
    parser.add_argument("--n-threads", type=int, default=None, help="Hilos de CPU a usar (default: autodetectado por llama.cpp).")
    parser.add_argument("--n-ctx", type=int, default=DEFAULT_CONTEXT_WINDOW, help="Ventana de contexto (default: 4096).")
    parser.add_argument("--json", action="store_true", help="Imprime el resultado como JSON en vez de texto legible.")
    parser.add_argument(
        "--output", default=str(DEFAULT_RESULT_PATH),
        help=f"Ruta donde persistir el resultado como JSON (default: {DEFAULT_RESULT_PATH}). "
             "scripts/generate_plots.py lee este archivo si existe.",
    )
    args = parser.parse_args()

    server = LLMServer(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=0,
        n_threads=args.n_threads,
    )
    config = GenerationConfig(max_tokens=args.max_tokens)
    result = run_benchmark(server, args.prompt, config=config)

    payload = {
        "model_path": str(args.model_path),
        "model_label": args.model_label or Path(args.model_path).stem,
        "n_threads": args.n_threads,
        "max_tokens": args.max_tokens,
        **result.__dict__,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_module.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json_module.dumps(payload, indent=2))
        print(f"\n[guardado en {output_path}]")
        return

    print(f"Modelo:              {args.model_path}")
    print(f"Tokens generados:    {result.tokens_generated}")
    print(f"TTFT:                {result.ttft_seconds * 1000:.1f} ms")
    print(f"Throughput:          {result.tokens_per_second:.2f} tok/s")
    print(f"Tiempo total:        {result.total_seconds:.2f} s")
    print(f"RAM (RSS):           {result.ram_mb:.1f} MB" if result.ram_mb is not None else "RAM (RSS):           no disponible (psutil ausente)")
    print(f"VRAM:                {result.vram_mb:.1f} MB" if result.vram_mb is not None else "VRAM:                N/A (sin GPU NVIDIA / forzado a CPU)")
    print(f"\n[guardado en {output_path}]")


if __name__ == "__main__":  # pragma: no cover
    _run_cli()
