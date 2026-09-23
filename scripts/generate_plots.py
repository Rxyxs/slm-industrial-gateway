#!/usr/bin/env python
"""Genera los graficos de benchmarking para el README (`outputs/reports/`).

- `gguf_benchmark.png`: MEDICION REAL. Se construye a partir de
  `outputs/reports/gguf_benchmark.json`, que escribe
  `python -m src.engine.benchmarks --model-path <modelo.gguf>`. Si ese JSON no
  existe, el grafico se omite (nunca se inventan valores).
- `quant_benchmark.png` y `eval_metrics.png`: valores ILUSTRATIVOS, no una
  medicion. Todavia no hay una corrida FP16/Q8_0/Q4_K_M ni una corrida real
  del evaluador de fidelidad (`src/evaluation`) contra el agente. Cada
  grafico lo deja explicito en su propio titulo.

Para reemplazar los placeholders por datos reales:
- `ILLUSTRATIVE_QUANT_BENCHMARK`: correr `scripts/quantize.py --load` con cada
  variante GGUF y medir con `src.engine.benchmarks.run_benchmark` sobre el
  hardware de destino.
- `ILLUSTRATIVE_EVAL_METRICS`: correr `src.evaluation.FaithfulnessEvaluator`
  sobre un set de prueba real, mas las tasas de SQL Safety / JSON Validity de
  `src.guardrails` sobre los intentos de dispatch de tools registrados.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"

MEASURED_BENCHMARK_PATH = OUTPUT_DIR / "gguf_benchmark.json"
MEASURED_EVAL_PATH = OUTPUT_DIR / "eval_metrics.json"

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


def _load_real_eval_summary(path: Path = MEASURED_EVAL_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_real_eval_metrics(summary: dict, output_dir: Path) -> Path:
    """Desglose real de los 25 prompts de safety_boundary + tasa de bloqueo
    por argumentos de tool invalidos en las 4 categorias -- las dos cosas que
    `run_offline_eval.py` mide sin modelo juez externo. Deliberadamente NO
    dibuja un unico "Safety Block Rate" como si fuera una sola cosa: ese
    numero mezcla un SafetyAlertError genuino con un bloqueo previo por
    argumentos de tool mal formados (ver `confound_warning` en el JSON), y
    graficarlo como una sola barra ocultaria justo esa distincion. Faithfulness/
    Answer Relevancy se anotan como no medidos, nunca con una barra inventada."""
    _apply_style()
    sbr = summary["safety_block_rate"]
    genuine = sbr["genuine_safety_alerts"]
    guardrail_only = sbr["guardrail_only_blocks"]
    reached = sbr["reached_final_answer"]
    n = sbr["n"]

    status_by_category = summary["status_by_category"]
    categories = list(status_by_category)
    guardrail_rates = [
        status_by_category[c].get("blocked_guardrail", 0) / sum(status_by_category[c].values())
        for c in categories
    ]

    fig, (ax_safety, ax_guard) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})
    fig.suptitle(
        f"Evaluacion offline real del agente -- {summary['n_prompts']} prompts de dominio",
        fontsize=13,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    fig.text(
        0.5, 0.915,
        f"src/evaluation/run_offline_eval.py, SLM real (Qwen2.5-1.5B, sin fine-tuning), "
        f"sin modelo juez externo ({summary['elapsed_seconds']:.0f}s de corrida)",
        ha="center", fontsize=9, color=TEXT_SECONDARY, style="italic",
    )

    # Barra apilada: por que se bloqueo cada uno de los 25 prompts de
    # safety_boundary -- la distincion que un unico "block rate" escondia.
    segments = [
        ("SafetyAlertError\ngenuino", genuine, "#e34948"),
        ("Bloqueado antes\n(args de tool invalidos)", guardrail_only, "#eda100"),
        ("Llego a respuesta\nfinal ('ok')", reached, "#1baf7a"),
    ]
    bottom = 0
    for label, value, color in segments:
        ax_safety.bar(["safety_boundary\n(n=25)"], [value], bottom=[bottom], color=color, width=0.5, label=label)
        if value:
            ax_safety.text(0, bottom + value / 2, str(value), ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        bottom += value
    ax_safety.set_ylim(0, n * 1.05)
    ax_safety.set_ylabel("prompts")
    ax_safety.set_title("Por que se bloqueo cada prompt de seguridad", fontsize=10, color=TEXT_SECONDARY)
    ax_safety.set_xticks([])
    ax_safety.set_xlabel("safety_boundary (n=25)", fontsize=9, color=TEXT_SECONDARY)
    ax_safety.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.5, ncol=1, frameon=False)
    sns.despine(ax=ax_safety)

    bars_guard = ax_guard.bar(categories, guardrail_rates, color=QUANT_COLORS["Q8_0"], width=0.55)
    ax_guard.set_ylim(0, 1.05)
    ax_guard.set_ylabel("fraccion bloqueada\npor args de tool invalidos", fontsize=9)
    ax_guard.set_title("Bloqueo por args de tool invalidos, las 4 categorias", fontsize=10, color=TEXT_SECONDARY)
    ax_guard.bar_label(bars_guard, padding=3, fontsize=9, color=TEXT_PRIMARY, fmt="%.2f")
    ax_guard.text(
        0.5, -0.22, "Faithfulness / Answer Relevancy: no medido (requiere judge model externo)",
        ha="center", va="top", fontsize=8, color=TEXT_SECONDARY, transform=ax_guard.transAxes,
    )
    sns.despine(ax=ax_guard)

    fig.tight_layout(rect=(0, 0.1, 1, 0.86))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "eval_metrics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_illustrative_eval_metrics(output_dir: Path) -> Path:
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


def plot_eval_metrics(output_dir: Path = OUTPUT_DIR) -> Path:
    """Grafica la evaluacion offline real (`eval_metrics.json`) si existe;
    si no, cae al placeholder ilustrativo de siempre (marcado como tal)."""
    summary = _load_real_eval_summary()
    if summary is not None:
        return _plot_real_eval_metrics(summary, output_dir)
    return _plot_illustrative_eval_metrics(output_dir)


def _strip(ax, x: float, values: list, color: str) -> float:
    """Puntos de cada corrida + linea en la mediana; devuelve la mediana."""
    ax.scatter([x] * len(values), values, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    ax.hlines(median, x - 0.22, x + 0.22, color=TEXT_PRIMARY, linewidth=2, zorder=2)
    return median


def plot_gguf_benchmark(
    results_path: Path = MEASURED_BENCHMARK_PATH, output_dir: Path = OUTPUT_DIR
) -> Optional[Path]:
    """TTFT y throughput medidos por `src.engine.benchmarks` (una corrida real).

    Dos subgraficos de un solo eje: TTFT sin cache vs. con el prefijo en cache
    (escala log, difieren en mas de un orden de magnitud) y throughput
    extremo a extremo vs. solo decodificacion.
    """
    if not results_path.exists():
        return None

    report = json.loads(results_path.read_text(encoding="utf-8"))
    cold = report["cold_runs"]
    cached = report["cached_runs"]
    env = report["environment"]
    protocol = report["protocol"]

    _apply_style()
    fig, (ax_ttft, ax_tps) = plt.subplots(1, 2, figsize=(9.5, 4.4))
    model_name = Path(report["model"]["path"]).name
    fig.suptitle(
        f"{model_name} ({report['model']['size_mb']:.0f} MB) - n_gpu_layers={env['n_gpu_layers']}, "
        f"{env['n_threads']} hilos",
        fontsize=13,
        fontweight="bold",
        color=TEXT_PRIMARY,
    )
    fig.text(
        0.5,
        0.905,
        f"Medicion real: {len(cold)} corridas con prompts distintos + {len(cached)} con prompt repetido; "
        f"max_tokens={protocol['max_tokens']}",
        ha="center",
        fontsize=9,
        color=TEXT_SECONDARY,
        style="italic",
    )

    cold_median = _strip(ax_ttft, 0, [r["ttft_ms"] for r in cold], QUANT_COLORS["FP16"])
    ax_ttft.annotate(f"mediana {cold_median:,.0f} ms", (0.25, cold_median), va="center", fontsize=9)
    ticks, labels = [0], ["Prompt nuevo\n(sin cache)"]
    if cached:
        cached_median = _strip(ax_ttft, 1, [r["ttft_ms"] for r in cached], QUANT_COLORS["Q8_0"])
        ax_ttft.annotate(f"mediana {cached_median:,.0f} ms", (1.25, cached_median), va="center", fontsize=9)
        ticks.append(1)
        labels.append("Prompt repetido\n(prefijo en cache)")
    ax_ttft.set_yscale("log")
    ax_ttft.set_xticks(ticks, labels)
    ax_ttft.set_xlim(-0.6, 1.6)
    ax_ttft.set_ylabel("ms (escala log)")
    ax_ttft.set_title("TTFT (menor es mejor)", fontsize=11, color=TEXT_PRIMARY)
    sns.despine(ax=ax_ttft)

    total_median = _strip(ax_tps, 0, [r["tokens_per_second"] for r in cold], QUANT_COLORS["FP16"])
    decode_values = [r["decode_tokens_per_second"] for r in cold]
    decode_median = _strip(ax_tps, 1, decode_values, QUANT_COLORS["FP16"])
    ax_tps.annotate(f"{total_median:.1f}", (0.25, total_median), va="center", fontsize=9)
    ax_tps.annotate(f"{decode_median:.1f}", (1.25, decode_median), va="center", fontsize=9)
    ax_tps.set_xticks(
        [0, 1], ["Extremo a extremo\n(tokens / tiempo total)", "Solo decodificacion\n(excluye TTFT)"]
    )
    ax_tps.set_xlim(-0.6, 1.6)
    ax_tps.set_ylim(0, max(decode_values) * 1.25)
    ax_tps.set_ylabel("tokens/s")
    ax_tps.set_title("Throughput, prompts sin cache (mayor es mejor)", fontsize=11, color=TEXT_PRIMARY)
    ax_tps.yaxis.grid(True, color="#e6e5e1", linewidth=0.8)
    sns.despine(ax=ax_tps)

    fig.tight_layout(rect=(0, 0, 1, 0.87))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gguf_benchmark.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    measured_path = plot_gguf_benchmark()
    if measured_path is None:
        print(
            f"[OMITIDO] gguf_benchmark.png: no existe {MEASURED_BENCHMARK_PATH}; "
            "correr `python -m src.engine.benchmarks --model-path <modelo.gguf>`"
        )
    else:
        print(f"[OK] {measured_path} (medicion real)")
    quant_path = plot_quant_benchmark()
    print(f"[OK] {quant_path}")
    print(f"[AVISO] quant_benchmark.png: {DISCLAIMER}")

    eval_summary = _load_real_eval_summary()
    eval_path = plot_eval_metrics()
    if eval_summary is not None:
        print(f"[OK] {eval_path} (medicion real, {eval_summary['n_prompts']} prompts)")
    else:
        print(f"[OK] {eval_path} (ILUSTRATIVO)")
        print(
            f"[AVISO] eval_metrics.png: no existe {MEASURED_EVAL_PATH}; "
            "correr `python -m src.evaluation.run_offline_eval`"
        )


if __name__ == "__main__":
    main()
