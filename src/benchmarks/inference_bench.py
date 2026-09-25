"""Arnés de benchmarking de inferencia: barre `n_ctx` y `n_threads`, y para
cada combinación mide TTFT, throughput de prefill y de decodificación, y
memoria (RSS/VMS) -- en frío (prefijo nuevo) y en caliente (mismo prefijo
reutilizado).

Distinto de `src/engine/benchmarks.py::run_suite`, que mide una única
configuración fija del motor en profundidad (varias corridas, un control de
caché, para reportar mediana/mín/máx): este módulo REUSA esos primitivos
(`run_benchmark`, `reset_prompt_cache`, `decode_tokens_per_second`,
`current_ram_mb`/`current_vram_mb`) para en cambio recorrer **varias
configuraciones** del motor y comparar el efecto de `n_ctx`/`n_threads` entre
sí -- una corrida rápida por punto, no una batería estadística por punto.

Uso:

    python -m src.benchmarks.inference_bench --model-path data/models/model.gguf
    python -m src.benchmarks.inference_bench --dry-run  # sin .gguf real, para CI
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    import psutil
except ImportError:  # pragma: no cover - opcional, degrada a None
    psutil = None  # type: ignore[assignment]

from src.engine.benchmarks import (
    BenchmarkResult,
    build_benchmark_prompts,
    current_ram_mb,
    current_vram_mb,
    decode_tokens_per_second,
    reset_prompt_cache,
    run_benchmark,
)
from src.engine.llm_server import GenerationConfig, LLMServer

DEFAULT_N_CTX_VALUES = (1024, 2048, 4096)
DEFAULT_N_THREADS_VALUES = (2, 4)
DEFAULT_MAX_TOKENS = 32
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "benchmarks" / "inference_benchmark.json"


class _DryRunLLM:
    """Sustituto mínimo de `llama_cpp.Llama` para `_DryRunEngine`: solo lo
    que `LLMServer` toca (`tokenize`, `reset`) -- para que `reset_prompt_cache`
    (que espera un `server._llm.reset()`) funcione igual con el motor
    simulado, sin duplicar esa función."""

    def reset(self) -> None:
        pass

    def tokenize(self, data: bytes) -> list[int]:
        return list(range(max(len(data.split()), 1)))


class _DryRunEngine:
    """Motor simulado para `--dry-run`: mismo contrato que `LLMServer` en lo
    que este módulo necesita (`generate_stream`, `count_tokens`, `close`,
    `_llm.reset()`), sin cargar ningún `.gguf` real -- así el arnés completo
    (incluida la matemática de TTFT/tokens-por-segundo sobre timings reales,
    aunque sintéticos) corre en CI, donde no hay pesos pesados disponibles.

    Los retrasos son deliberadamente pequeños (milisegundos) para que la
    suite de tests siga siendo rápida, pero > 0 -- así se ejercita la misma
    división por tiempo transcurrido que usaría un modelo real, en vez de
    devolver ceros que no probarían nada.
    """

    def __init__(self, n_ctx: int, n_threads: Optional[int] = None,
                 prefill_seconds_per_token: float = 0.0005, decode_seconds_per_token: float = 0.002):
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = 0
        self._llm = _DryRunLLM()
        self._prefill_seconds_per_token = prefill_seconds_per_token
        self._decode_seconds_per_token = decode_seconds_per_token

    def count_tokens(self, text: str) -> int:
        return self._llm.tokenize(text.encode("utf-8")).__len__()

    def generate_stream(self, prompt: str, config: Optional[GenerationConfig] = None):
        cfg = config or GenerationConfig()
        prompt_tokens = self.count_tokens(prompt)
        time.sleep(prompt_tokens * self._prefill_seconds_per_token)  # simula el prefill
        for _ in range(cfg.max_tokens):
            time.sleep(self._decode_seconds_per_token)  # simula un paso de decodificación
            yield "tok "

    def close(self) -> None:
        pass


def prompt_processing_tokens_per_second(prompt_tokens: int, ttft_seconds: float) -> float:
    """Tokens/segundo de la fase de prefill (ingesta del prompt): tokens del
    prompt divididos por TTFT. `0.0` (no una excepción) si TTFT es cero o
    negativo o no hay tokens de prompt -- el caso de muestras ultra-rápidas
    que decidí manejar explícitamente sin dividir por cero.
    """
    if ttft_seconds <= 0 or prompt_tokens <= 0:
        return 0.0
    return prompt_tokens / ttft_seconds


def current_memory_mb() -> dict[str, Optional[float]]:
    """RSS y VMS actuales del proceso, en MB. `None` en ambos si `psutil` no
    está instalado (igual que `current_ram_mb` en `src/engine/benchmarks.py`,
    que solo expone RSS)."""
    if psutil is None:
        return {"rss_mb": None, "vms_mb": None}
    mem = psutil.Process().memory_info()
    return {"rss_mb": mem.rss / (1024 * 1024), "vms_mb": mem.vms / (1024 * 1024)}


def _build_engine(model_path: Optional[Path], n_ctx: int, n_threads: Optional[int], dry_run: bool):
    if dry_run:
        return _DryRunEngine(n_ctx=n_ctx, n_threads=n_threads)
    if model_path is None:
        raise ValueError("--model-path es obligatorio salvo que se use --dry-run.")
    return LLMServer(model_path, n_ctx=n_ctx, n_threads=n_threads)


def _result_to_point(result: BenchmarkResult, prompt_tokens: int) -> dict[str, Any]:
    return {
        "ttft_ms": result.ttft_seconds * 1000,
        "generation_tokens_per_sec": decode_tokens_per_second(result),
        "prompt_processing_tokens_per_sec": prompt_processing_tokens_per_second(
            prompt_tokens, result.ttft_seconds
        ),
        "prompt_tokens": prompt_tokens,
        "tokens_generated": result.tokens_generated,
        "total_seconds": result.total_seconds,
        "ram_mb": result.ram_mb,
        "vram_mb": result.vram_mb,
    }


@dataclass
class SweepConfig:
    n_ctx_values: Sequence[int] = DEFAULT_N_CTX_VALUES
    n_threads_values: Sequence[Optional[int]] = DEFAULT_N_THREADS_VALUES
    max_tokens: int = DEFAULT_MAX_TOKENS
    dry_run: bool = False
    model_path: Optional[Path] = None


def run_sweep(sweep: SweepConfig) -> dict[str, Any]:
    """Recorre cada combinación de `n_ctx` × `n_threads`: por cada una, carga
    (o simula) el motor una vez, y mide un punto en frío (prefijo nuevo,
    `reset_prompt_cache`) y uno en caliente (mismo prompt, prefijo ya en
    caché) -- el arranque frío/cálido, reusando el mismo
    prompt de diagnóstico industrial que usa `run_suite`.
    """
    config = GenerationConfig(max_tokens=sweep.max_tokens, temperature=0.0)
    prompt = build_benchmark_prompts(1)[0]

    points: list[dict[str, Any]] = []
    peak_rss_mb = 0.0
    peak_vms_mb = 0.0

    for n_ctx in sweep.n_ctx_values:
        for n_threads in sweep.n_threads_values:
            engine = _build_engine(sweep.model_path, n_ctx, n_threads, sweep.dry_run)
            try:
                prompt_tokens = engine.count_tokens(prompt)

                reset_prompt_cache(engine)
                cold = run_benchmark(engine, prompt, config)

                warm = run_benchmark(engine, prompt, config)  # mismo prompt, sin reset: prefijo en cache

                mem = current_memory_mb()
                if mem["rss_mb"] is not None:
                    peak_rss_mb = max(peak_rss_mb, mem["rss_mb"])
                if mem["vms_mb"] is not None:
                    peak_vms_mb = max(peak_vms_mb, mem["vms_mb"])

                points.append({
                    "n_ctx": n_ctx,
                    "n_threads": n_threads,
                    "cold": _result_to_point(cold, prompt_tokens),
                    "warm": _result_to_point(warm, prompt_tokens),
                })
            finally:
                engine.close()

    return {
        "dry_run": sweep.dry_run,
        "max_tokens": sweep.max_tokens,
        "ram_mb": current_ram_mb(),
        "vram_mb": current_vram_mb(),
        "peak_rss_mb": peak_rss_mb if psutil is not None else None,
        "peak_vms_mb": peak_vms_mb if psutil is not None else None,
        "points": points,
    }


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmarks.inference_bench",
        description="Barrido de n_ctx/n_threads midiendo TTFT, throughput de prefill/decodificación y memoria.",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="Ruta al modelo .gguf (ignorado con --dry-run).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Usa un motor simulado en vez de llama.cpp -- para CI, sin .gguf real.",
    )
    parser.add_argument("--n-ctx-values", type=int, nargs="+", default=list(DEFAULT_N_CTX_VALUES))
    parser.add_argument("--n-threads-values", type=int, nargs="+", default=list(DEFAULT_N_THREADS_VALUES))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--output-json", type=Path, nargs="?", const=DEFAULT_OUTPUT_PATH, default=None,
        help=f"Guarda el resultado detallado en JSON (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if not args.dry_run and args.model_path is None:
        print("ERROR: --model-path es obligatorio salvo que se use --dry-run.", file=sys.stderr)
        return 1

    sweep = SweepConfig(
        n_ctx_values=args.n_ctx_values,
        n_threads_values=args.n_threads_values,
        max_tokens=args.max_tokens,
        dry_run=args.dry_run,
        model_path=args.model_path,
    )
    resultado = run_sweep(sweep)

    print(f"Barrido: n_ctx={sweep.n_ctx_values}  n_threads={sweep.n_threads_values}  dry_run={sweep.dry_run}")
    for punto in resultado["points"]:
        print(
            f"  n_ctx={punto['n_ctx']:<5} n_threads={str(punto['n_threads']):<5} "
            f"TTFT frio={punto['cold']['ttft_ms']:.1f}ms  "
            f"prefill={punto['cold']['prompt_processing_tokens_per_sec']:.1f}tok/s  "
            f"decode={punto['cold']['generation_tokens_per_sec']:.1f}tok/s  "
            f"TTFT calido={punto['warm']['ttft_ms']:.1f}ms"
        )
    print(f"RSS pico: {resultado['peak_rss_mb']}  VMS pico: {resultado['peak_vms_mb']}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Resultados -> {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
