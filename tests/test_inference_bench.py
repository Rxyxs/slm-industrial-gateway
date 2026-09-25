"""Pruebas del arnés de benchmarking de inferencia (src/benchmarks/inference_bench.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarks.inference_bench import (
    SweepConfig,
    _DryRunEngine,
    current_memory_mb,
    main,
    prompt_processing_tokens_per_second,
    run_sweep,
)
from src.engine.benchmarks import BenchmarkResult, decode_tokens_per_second

REQUIRED_POINT_FIELDS = {
    "ttft_ms", "generation_tokens_per_sec", "prompt_processing_tokens_per_sec",
    "prompt_tokens", "tokens_generated", "total_seconds", "ram_mb", "vram_mb",
}


# ---------------------------------------------------------------------------
# matemática: TTFT / prefill / decode, sin división por cero
# ---------------------------------------------------------------------------

def test_prompt_processing_tokens_per_second_is_exact():
    # 100 tokens de prompt procesados en 0.5s exactos -> 200 tok/s exacto.
    assert prompt_processing_tokens_per_second(100, 0.5) == pytest.approx(200.0)


def test_prompt_processing_tokens_per_second_is_zero_when_ttft_is_zero():
    """Muestra ultra-rápida: TTFT medido en 0.0s (posible con un mock/reloj
    de baja resolución) no debe dividir por cero."""
    assert prompt_processing_tokens_per_second(50, 0.0) == 0.0


def test_prompt_processing_tokens_per_second_is_zero_with_negative_ttft():
    """No debería ocurrir con un reloj monotónico real, pero si ocurriera
    (por ejemplo con un mock mal construido) tampoco debe lanzar."""
    assert prompt_processing_tokens_per_second(50, -0.01) == 0.0


def test_prompt_processing_tokens_per_second_is_zero_without_prompt_tokens():
    assert prompt_processing_tokens_per_second(0, 0.5) == 0.0


def test_decode_tokens_per_second_handles_a_single_generated_token():
    """Reexportado de src.engine.benchmarks, pero se verifica acá porque el
    arnés de sweep depende de este guard contra división por cero tanto como
    de prompt_processing_tokens_per_second."""
    resultado = BenchmarkResult(
        ttft_seconds=0.01, total_seconds=0.01, tokens_generated=1,
        tokens_per_second=100.0, ram_mb=None, vram_mb=None,
    )
    # Con 1 solo token generado no hay fase de decodificación medible
    # (el único token generado ES el TTFT) -- 0.0, no una excepción.
    assert decode_tokens_per_second(resultado) == 0.0


# ---------------------------------------------------------------------------
# _DryRunEngine: contrato compatible con LLMServer
# ---------------------------------------------------------------------------

def test_dry_run_engine_generate_stream_yields_max_tokens_chunks():
    from src.engine.llm_server import GenerationConfig

    engine = _DryRunEngine(n_ctx=1024, prefill_seconds_per_token=0.0, decode_seconds_per_token=0.0)
    chunks = list(engine.generate_stream("hola mundo", config=GenerationConfig(max_tokens=7)))

    assert len(chunks) == 7


def test_dry_run_engine_count_tokens_is_deterministic():
    engine = _DryRunEngine(n_ctx=1024)
    assert engine.count_tokens("uno dos tres") == 3


def test_dry_run_engine_reset_prompt_cache_does_not_raise():
    from src.engine.benchmarks import reset_prompt_cache

    engine = _DryRunEngine(n_ctx=1024)
    reset_prompt_cache(engine)  # no debe lanzar: engine._llm.reset() existe


# ---------------------------------------------------------------------------
# current_memory_mb
# ---------------------------------------------------------------------------

def test_current_memory_mb_returns_rss_and_vms():
    memoria = current_memory_mb()
    assert "rss_mb" in memoria and "vms_mb" in memoria
    if memoria["rss_mb"] is not None:  # psutil disponible en este entorno
        assert memoria["rss_mb"] > 0
        assert memoria["vms_mb"] > 0


# ---------------------------------------------------------------------------
# run_sweep con --dry-run: estructura JSON completa
# ---------------------------------------------------------------------------

def test_run_sweep_dry_run_produces_one_point_per_combination():
    sweep = SweepConfig(
        n_ctx_values=[1024, 2048], n_threads_values=[2, 4], max_tokens=3, dry_run=True,
    )
    resultado = run_sweep(sweep)

    assert len(resultado["points"]) == 4  # 2 n_ctx x 2 n_threads
    combinaciones = {(p["n_ctx"], p["n_threads"]) for p in resultado["points"]}
    assert combinaciones == {(1024, 2), (1024, 4), (2048, 2), (2048, 4)}


def test_run_sweep_dry_run_each_point_has_all_required_fields():
    sweep = SweepConfig(n_ctx_values=[1024], n_threads_values=[2], max_tokens=3, dry_run=True)
    resultado = run_sweep(sweep)

    punto = resultado["points"][0]
    assert set(punto.keys()) == {"n_ctx", "n_threads", "cold", "warm"}
    assert REQUIRED_POINT_FIELDS <= punto["cold"].keys()
    assert REQUIRED_POINT_FIELDS <= punto["warm"].keys()


def test_run_sweep_dry_run_reports_positive_token_throughput():
    sweep = SweepConfig(n_ctx_values=[1024], n_threads_values=[2], max_tokens=5, dry_run=True)
    resultado = run_sweep(sweep)

    punto = resultado["points"][0]
    assert punto["cold"]["generation_tokens_per_sec"] > 0
    assert punto["cold"]["prompt_processing_tokens_per_sec"] > 0
    assert punto["cold"]["tokens_generated"] == 5


def test_run_sweep_without_model_path_and_without_dry_run_raises():
    sweep = SweepConfig(n_ctx_values=[1024], n_threads_values=[2], dry_run=False, model_path=None)
    with pytest.raises(ValueError, match="--model-path"):
        run_sweep(sweep)


# ---------------------------------------------------------------------------
# CLI: --dry-run, --output-json
# ---------------------------------------------------------------------------

def test_cli_dry_run_exits_zero(capsys):
    codigo = main(["--dry-run", "--n-ctx-values", "1024", "--n-threads-values", "2", "--max-tokens", "3"])
    assert codigo == 0


def test_cli_without_model_path_and_without_dry_run_exits_nonzero(capsys):
    codigo = main(["--n-ctx-values", "1024"])
    assert codigo != 0


def test_cli_output_json_writes_a_valid_json_file_with_all_metrics(tmp_path):
    destino = tmp_path / "sub" / "inference_benchmark.json"
    codigo = main([
        "--dry-run", "--n-ctx-values", "1024", "2048", "--n-threads-values", "2",
        "--max-tokens", "3", "--output-json", str(destino),
    ])

    assert codigo == 0
    assert destino.exists()
    contenido = json.loads(destino.read_text(encoding="utf-8"))
    assert contenido["dry_run"] is True
    assert len(contenido["points"]) == 2
    for punto in contenido["points"]:
        assert REQUIRED_POINT_FIELDS <= punto["cold"].keys()


def test_cli_output_json_bare_flag_uses_the_default_path(monkeypatch, tmp_path):
    """`--output-json` sin argumento (el flag solo) tiene que usar
    DEFAULT_OUTPUT_PATH -- se le hace monkeypatch a un tmp_path para no
    escribir en el repo real durante los tests. `_parse_args` resuelve
    `DEFAULT_OUTPUT_PATH` como global del módulo en cada llamada, así que el
    patch aplica igual aunque `argparse` ya haya sido configurado antes."""
    import src.benchmarks.inference_bench as mod

    destino_default = tmp_path / "inference_benchmark.json"
    monkeypatch.setattr(mod, "DEFAULT_OUTPUT_PATH", destino_default)

    codigo = main([
        "--dry-run", "--n-ctx-values", "1024", "--n-threads-values", "2",
        "--max-tokens", "3", "--output-json",  # bandera sola, sin argumento
    ])

    assert codigo == 0
    assert destino_default.exists()


def test_cli_without_output_json_does_not_write_any_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    codigo = main(["--dry-run", "--n-ctx-values", "1024", "--n-threads-values", "2", "--max-tokens", "3"])

    assert codigo == 0
    assert not (tmp_path / "outputs").exists()
