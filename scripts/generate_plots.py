#!/usr/bin/env python
"""Genera los graficos de benchmarking para el README (`outputs/reports/`).

IMPORTANTE - los valores de este script son ILUSTRATIVOS, no una medicion
real: no hay GPU ni un modelo GGUF cargado en el entorno donde se escribio
este script, por lo que no existe todavia una corrida real de
`src/engine/benchmarks.run_benchmark` (FP16/Q8_0/Q4_K_M) ni una corrida real
del evaluador de fidelidad (`src/evaluation`) contra el agente. Cada grafico
lo deja explicito en su propio titulo.

Para reemplazar los placeholders por datos reales:
- `ILLUSTRATIVE_QUANT_BENCHMARK`: correr `scripts/quantize.py --load` con cada
  variante GGUF y medir con `src.engine.benchmarks.run_benchmark` sobre el
  hardware de destino.
- `ILLUSTRATIVE_EVAL_METRICS`: correr `src.evaluation.FaithfulnessEvaluator`
  sobre un set de prueba real, mas las tasas de SQL Safety / JSON Validity de
  `src.guardrails` sobre los intentos de dispatch de tools registrados.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"

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


def plot_quant_benchmark(output_dir: Path = OUTPUT_DIR) -> Path:
    """TTFT (ms) y throughput (tok/s) por nivel de cuantizacion GGUF.

    Dos subgraficos con un solo eje Y cada uno (nunca doble eje en el mismo
    grafico): TTFT y throughput tienen escalas distintas y no son comparables
    en una misma vara de medida.
    """
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
    quant_path = plot_quant_benchmark()
    eval_path = plot_eval_metrics()
    print(f"[OK] {quant_path}")
    print(f"[OK] {eval_path}")
    print(f"[AVISO] {DISCLAIMER}")


if __name__ == "__main__":
    main()
