#!/usr/bin/env python
"""Genera los graficos de benchmarking para el README (`outputs/reports/`).

El grafico de cuantizacion (`plot_quant_benchmark`) lee, si existe,
`outputs/reports/benchmark_result.json` -- el archivo que escribe
`python -m src.engine.benchmarks` en cada corrida real -- y grafica esos
numeros medidos, sin ILUSTRATIVO en el titulo. Si el archivo no existe (nadie
corrio el benchmark todavia en este checkout), cae al placeholder de siempre
y lo marca como tal: el script nunca finge tener una medicion que no existe.

El grafico de metricas de evaluacion (`plot_eval_metrics`) SIGUE siendo
ilustrativo: correr `src.evaluation.FaithfulnessEvaluator` de verdad requiere
un modelo juez externo (por defecto gpt-4o-mini via API), que este comando no
invoca. Reemplazar `ILLUSTRATIVE_EVAL_METRICS` exige correrlo aparte sobre un
set de prueba real, mas las tasas de SQL Safety / JSON Validity de
`src.guardrails` sobre intentos de dispatch de tools registrados.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"
BENCHMARK_RESULT_PATH = OUTPUT_DIR / "benchmark_result.json"

DISCLAIMER = "Datos ilustrativos - pendientes de validar en hardware real"

# Paleta categorica de referencia (azul / naranja / aqua): los tres primeros
# slots del tema por defecto, los unicos que validan todos-contra-todos.
QUANT_COLORS = {"FP16": "#2a78d6", "Q8_0": "#eb6834", "Q4_K_M": "#1baf7a"}
EVAL_COLOR = "#2a78d6"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

# Placeholders: reemplazar por resultados reales antes de presentarlos como medicion.
ILLUSTRATIVE_QUANT_BENCHMARK = {
    "FP16": {"ttft_ms": 180, "throughput_tps": 22},
    "Q8_0": {"ttft_ms": 95, "throughput_tps": 38},
    "Q4_K_M": {"ttft_ms": 60, "throughput_tps": 54},
}

ILLUSTRATIVE_EVAL_METRICS = {
    "Faithfulness": 0.93,
    "Answer\nRelevancy": 0.89,
    "SQL Safety\nRate": 1.00,
    "JSON\nValidity": 0.97,
}


def _apply_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "xtick.color": TEXT_PRIMARY,
            "ytick.color": TEXT_PRIMARY,
            "axes.edgecolor": TEXT_SECONDARY,
            "font.size": 11,
        }
    )


def _load_real_benchmark(path: Path = BENCHMARK_RESULT_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_real_quant_benchmark(real: dict, output_dir: Path) -> Path:
    """TTFT, throughput y RAM de la corrida real guardada por
    `src.engine.benchmarks` -- un solo modelo/cuantizacion (el que se haya
    benchmarkeado), no una comparacion de tres niveles: no hay datos reales
    de FP16/Q8_0 en CPU para comparar contra Q4_K_M en esta corrida."""
    _apply_style()
    label = real.get("model_label") or Path(real["model_path"]).stem
    color = QUANT_COLORS["Q4_K_M"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    fig.suptitle(
        f"Benchmark real de inferencia en CPU -- {label}",
        fontsize=13,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    fig.text(
        0.5, 0.90, "Medido con src/engine/benchmarks.py -- ver outputs/reports/benchmark_result.json",
        ha="center", fontsize=9, color=TEXT_SECONDARY, style="italic",
    )

    ax_ttft, ax_tps, ax_ram = axes
    ttft_ms = real["ttft_seconds"] * 1000
    bars_ttft = ax_ttft.bar([label], [ttft_ms], color=color, width=0.5)
    ax_ttft.set_title("TTFT en ms (menor es mejor)", fontsize=11, color=TEXT_PRIMARY)
    ax_ttft.set_ylabel("ms")
    ax_ttft.bar_label(bars_ttft, padding=3, fontsize=9, color=TEXT_PRIMARY, fmt="%.0f")
    sns.despine(ax=ax_ttft)

    bars_tps = ax_tps.bar([label], [real["tokens_per_second"]], color=color, width=0.5)
    ax_tps.set_title("Throughput en tok/s (mayor es mejor)", fontsize=11, color=TEXT_PRIMARY)
    ax_tps.set_ylabel("tokens/s")
    ax_tps.bar_label(bars_tps, padding=3, fontsize=9, color=TEXT_PRIMARY, fmt="%.2f")
    sns.despine(ax=ax_tps)

    ram_mb = real.get("ram_mb")
    bars_ram = ax_ram.bar([label], [ram_mb or 0], color=color, width=0.5)
    ax_ram.set_title("RAM residente en MB", fontsize=11, color=TEXT_PRIMARY)
    ax_ram.set_ylabel("MB")
    ax_ram.bar_label(bars_ram, padding=3, fontsize=9, color=TEXT_PRIMARY, fmt="%.0f")
    sns.despine(ax=ax_ram)

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "quant_benchmark.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_illustrative_quant_benchmark(output_dir: Path) -> Path:
    _apply_style()
    levels = list(ILLUSTRATIVE_QUANT_BENCHMARK)
    ttft = [ILLUSTRATIVE_QUANT_BENCHMARK[level]["ttft_ms"] for level in levels]
    throughput = [ILLUSTRATIVE_QUANT_BENCHMARK[level]["throughput_tps"] for level in levels]
    colors = [QUANT_COLORS[level] for level in levels]

    fig, (ax_ttft, ax_tps) = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.suptitle(
        "Cuantizacion GGUF: TTFT vs. throughput (ILUSTRATIVO)",
        fontsize=13,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    fig.text(0.5, 0.90, DISCLAIMER, ha="center", fontsize=9, color=TEXT_SECONDARY, style="italic")

    bars_ttft = ax_ttft.bar(levels, ttft, color=colors, width=0.6)
    ax_ttft.set_title("TTFT en ms (menor es mejor)", fontsize=11, color=TEXT_PRIMARY)
    ax_ttft.set_ylabel("ms")
    ax_ttft.bar_label(bars_ttft, padding=3, fontsize=9, color=TEXT_PRIMARY)
    sns.despine(ax=ax_ttft)

    bars_tps = ax_tps.bar(levels, throughput, color=colors, width=0.6)
    ax_tps.set_title("Throughput en tok/s (mayor es mejor)", fontsize=11, color=TEXT_PRIMARY)
    ax_tps.set_ylabel("tokens/s")
    ax_tps.bar_label(bars_tps, padding=3, fontsize=9, color=TEXT_PRIMARY)
    sns.despine(ax=ax_tps)

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "quant_benchmark.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_quant_benchmark(output_dir: Path = OUTPUT_DIR) -> Path:
    """Grafica la corrida real guardada en `benchmark_result.json` si existe;
    si no, cae al placeholder ilustrativo de 3 niveles (marcado como tal)."""
    real = _load_real_benchmark()
    if real is not None:
        return _plot_real_quant_benchmark(real, output_dir)
    return _plot_illustrative_quant_benchmark(output_dir)


def plot_eval_metrics(output_dir: Path = OUTPUT_DIR) -> Path:
    """Metricas de evaluacion del agente (una sola serie, escala 0-1 compartida)."""
    _apply_style()
    metrics = list(ILLUSTRATIVE_EVAL_METRICS)
    scores = [ILLUSTRATIVE_EVAL_METRICS[metric] for metric in metrics]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle(
        "Metricas de evaluacion del agente (ILUSTRATIVO)",
        fontsize=13,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    fig.text(0.5, 0.90, DISCLAIMER, ha="center", fontsize=9, color=TEXT_SECONDARY, style="italic")

    bars = ax.bar(metrics, scores, color=EVAL_COLOR, width=0.55)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (0-1)")
    ax.bar_label(bars, padding=3, fontsize=9, color=TEXT_PRIMARY, fmt="%.2f")
    sns.despine(ax=ax)

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "eval_metrics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    is_real = _load_real_benchmark() is not None
    quant_path = plot_quant_benchmark()
    eval_path = plot_eval_metrics()
    print(f"[OK] {quant_path}" + ("" if is_real else " (ILUSTRATIVO)"))
    print(f"[OK] {eval_path} (ILUSTRATIVO)")
    if not is_real:
        print(f"[AVISO] Grafico de cuantizacion: {DISCLAIMER}")
    print(f"[AVISO] Grafico de metricas de evaluacion: {DISCLAIMER}")


if __name__ == "__main__":
    main()
