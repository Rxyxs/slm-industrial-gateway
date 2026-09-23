"""Medición de rendimiento del motor de inferencia: t/s, TTFT y memoria.

Uso por línea de comandos:

    python -m src.engine.benchmarks --model-path data/models/model.gguf

Protocolo de `run_suite`: una corrida de calentamiento descartada; luego
`runs` corridas con prompts distintos, limpiando antes de cada una el estado
de llama.cpp para que el TTFT incluya la evaluación completa del prompt; y un
control con el mismo prompt repetido sin limpiar, que mide el TTFT con el
prefijo ya en caché. Sin ese `reset()`, llama-cpp-python reutiliza el prefijo
común con el prompt anterior y el TTFT sale artificialmente bajo.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "reports" / "gguf_benchmark.json"

BENCHMARK_ASSETS = (
    "Molino SAG 01",
    "Camión de extracción CAT-797",
    "Bomba de pulpa P-204",
    "Chancador primario C-1",
    "Correa transportadora CV-3",
    "Compresor K-12",
)

# El activo va al comienzo para que los prompts diverjan desde el primer token.
BENCHMARK_PROMPT_TEMPLATE = (
    "{asset}: la vibración subió de 2.1 a 6.8 mm/s RMS en 48 horas y la temperatura "
    "del rodamiento está en 84 grados C. Explica en 5 puntos las causas probables y "
    "las acciones de mantenimiento recomendadas."
)


def build_benchmark_prompts(count: int) -> List[str]:
    """Devuelve `count` prompts distintos de diagnóstico industrial."""
    prompts = []
    for index in range(count):
        asset = BENCHMARK_ASSETS[index % len(BENCHMARK_ASSETS)]
        if index >= len(BENCHMARK_ASSETS):
            asset = f"{asset} (unidad {index // len(BENCHMARK_ASSETS) + 1})"
        prompts.append(BENCHMARK_PROMPT_TEMPLATE.format(asset=asset))
    return prompts


def decode_tokens_per_second(result: BenchmarkResult) -> float:
    """Tokens/segundo de la fase de decodificación, excluyendo el TTFT.

    `BenchmarkResult.tokens_per_second` divide por el tiempo total, que
    incluye la evaluación del prompt; con respuestas cortas esa cifra
    subestima la velocidad de generación.
    """
    decode_seconds = result.total_seconds - result.ttft_seconds
    if result.tokens_generated < 2 or decode_seconds <= 0:
        return 0.0
    return (result.tokens_generated - 1) / decode_seconds


def reset_prompt_cache(server: LLMServer) -> None:
    """Limpia el estado de llama.cpp para que el próximo prompt se evalúe completo."""
    reset = getattr(getattr(server, "_llm", None), "reset", None)
    if callable(reset):
        reset()


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """Mediana, mínimo y máximo de una serie de mediciones."""
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def _run_to_dict(result: BenchmarkResult) -> Dict[str, Any]:
    row = asdict(result)
    row["ttft_ms"] = result.ttft_seconds * 1000
    row["decode_tokens_per_second"] = decode_tokens_per_second(result)
    return row


def run_suite(
    server: LLMServer,
    config: GenerationConfig,
    runs: int = 5,
    cached_repeats: int = 3,
    warmup: int = 1,
) -> Dict[str, Any]:
    """Ejecuta el protocolo completo (ver docstring del módulo) y devuelve las corridas y su resumen."""
    if runs < 1:
        raise ValueError("'runs' debe ser al menos 1.")

    prompts = build_benchmark_prompts(warmup + runs)
    for prompt in prompts[:warmup]:
        reset_prompt_cache(server)
        run_benchmark(server, prompt, config)

    cold: List[BenchmarkResult] = []
    for prompt in prompts[warmup:]:
        reset_prompt_cache(server)
        cold.append(run_benchmark(server, prompt, config))

    cached: List[BenchmarkResult] = []
    if cached_repeats > 0:
        cached_prompt = prompts[warmup]
        reset_prompt_cache(server)
        run_benchmark(server, cached_prompt, config)  # deja el prefijo en caché
        cached = [run_benchmark(server, cached_prompt, config) for _ in range(cached_repeats)]

    ram = [r.ram_mb for r in cold if r.ram_mb is not None]
    vram = [r.vram_mb for r in cold if r.vram_mb is not None]
    summary = {
        "ttft_ms": summarize([r.ttft_seconds * 1000 for r in cold]),
        "tokens_per_second": summarize([r.tokens_per_second for r in cold]),
        "decode_tokens_per_second": summarize([decode_tokens_per_second(r) for r in cold]),
        "tokens_generated": summarize([r.tokens_generated for r in cold]),
        "ram_mb": summarize(ram) if ram else None,
        "vram_mb": summarize(vram) if vram else None,
    }

    return {
        "cold_runs": [_run_to_dict(r) for r in cold],
        "cached_runs": [_run_to_dict(r) for r in cached],
        "summary": summary,
        "cached_summary": (
            {"ttft_ms": summarize([r.ttft_seconds * 1000 for r in cached])} if cached else None
        ),
    }


def _format_range(stats: Optional[Dict[str, float]], fmt: str) -> str:
    if stats is None:
        return "n/a"
    return f"{stats['median']:{fmt}} ({stats['min']:{fmt}} - {stats['max']:{fmt}})"


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.engine.benchmarks",
        description="Mide TTFT, throughput y memoria de un modelo GGUF local.",
    )
    parser.add_argument("--model-path", required=True, type=Path, help="Ruta al modelo .gguf.")
    parser.add_argument("--runs", type=int, default=5, help="Corridas medidas con prompts distintos.")
    parser.add_argument(
        "--cached-repeats", type=int, default=3,
        help="Corridas del control con el prefijo en caché (0 = omitir).",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-gpu-layers", type=int, default=None, help="Por defecto se autodetecta.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Archivo JSON de resultados.")
    return parser.parse_args(argv)


def _llama_cpp_version() -> Optional[str]:
    try:
        import llama_cpp
    except ImportError:  # pragma: no cover - LLMServer ya habría fallado
        return None
    return getattr(llama_cpp, "__version__", None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Carga el modelo, corre `run_suite`, imprime el resumen y guarda el JSON."""
    args = _parse_args(argv)
    config = GenerationConfig(max_tokens=args.max_tokens, temperature=0.0)

    ram_before_load = current_ram_mb()
    start = time.perf_counter()
    server = LLMServer(
        args.model_path, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers, n_threads=args.n_threads
    )
    load_seconds = time.perf_counter() - start
    ram_after_load = current_ram_mb()
    n_threads = getattr(getattr(server, "_llm", None), "n_threads", args.n_threads)

    try:
        suite = run_suite(server, config, runs=args.runs, cached_repeats=args.cached_repeats)
    finally:
        server.close()

    report = {
        "model": {
            "path": str(args.model_path),
            "size_mb": args.model_path.stat().st_size / (1024 * 1024),
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count() if psutil is not None else None,
            "llama_cpp_python": _llama_cpp_version(),
            "n_gpu_layers": server.n_gpu_layers,
            "n_threads": n_threads,
            "n_ctx": args.n_ctx,
        },
        "protocol": {
            "runs": args.runs,
            "cached_repeats": args.cached_repeats,
            "warmup": 1,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        },
        "load_seconds": load_seconds,
        "ram_before_load_mb": ram_before_load,
        "ram_after_load_mb": ram_after_load,
        **suite,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = suite["summary"]
    print(
        f"Modelo: {args.model_path} ({report['model']['size_mb']:.0f} MB), "
        f"n_gpu_layers={server.n_gpu_layers}, n_threads={n_threads}"
    )
    print(f"Corridas sin caché: {args.runs} -> mediana (mín - máx)")
    print(f"  TTFT (ms):                    {_format_range(summary['ttft_ms'], ',.0f')}")
    print(f"  Throughput total (tok/s):     {_format_range(summary['tokens_per_second'], '.2f')}")
    print(f"  Throughput decodif. (tok/s):  {_format_range(summary['decode_tokens_per_second'], '.2f')}")
    print(f"  RAM residente (MB):           {_format_range(summary['ram_mb'], ',.0f')}")
    print(f"  VRAM (MB):                    {_format_range(summary['vram_mb'], ',.0f')}")
    if suite["cached_summary"]:
        cached_ttft = _format_range(suite["cached_summary"]["ttft_ms"], ",.0f")
        print(f"Control con prefijo en caché, TTFT (ms): {cached_ttft}")
    print(f"Resultados: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
