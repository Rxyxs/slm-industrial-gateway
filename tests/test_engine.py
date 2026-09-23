"""Pruebas unitarias para el motor de inferencia (src/engine)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import benchmarks
from src.engine.benchmarks import (
    BenchmarkResult,
    build_benchmark_prompts,
    current_ram_mb,
    current_vram_mb,
    decode_tokens_per_second,
    run_benchmark,
    run_suite,
)
from src.engine.llm_server import (
    GenerationConfig,
    GenerationError,
    LLMServer,
    ModelLoadError,
    detect_gpu_layers,
)


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "model.Q4_K_M.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 16)
    return path


def make_completion(text: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [{"text": text, "index": 0, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": 1},
    }


# ---------------------------------------------------------------------------
# detect_gpu_layers
# ---------------------------------------------------------------------------


def test_detect_gpu_layers_respects_explicit_value():
    assert detect_gpu_layers(0) == 0
    assert detect_gpu_layers(-1) == -1
    assert detect_gpu_layers(20) == 20


def test_detect_gpu_layers_falls_back_to_cpu_without_nvidia_smi():
    with patch("src.engine.llm_server.shutil.which", return_value=None):
        assert detect_gpu_layers(None) == 0


def test_detect_gpu_layers_uses_gpu_when_nvidia_smi_present():
    fake_result = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 4090\n")
    with patch("src.engine.llm_server.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
        "src.engine.llm_server.subprocess.run", return_value=fake_result
    ):
        assert detect_gpu_layers(None) == -1


def test_detect_gpu_layers_handles_subprocess_failure():
    with patch("src.engine.llm_server.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
        "src.engine.llm_server.subprocess.run", side_effect=OSError("no permission")
    ):
        assert detect_gpu_layers(None) == 0


# ---------------------------------------------------------------------------
# LLMServer - carga del modelo
# ---------------------------------------------------------------------------


def test_llm_server_raises_if_llama_cpp_not_installed(model_file: Path):
    with patch("src.engine.llm_server.Llama", None):
        with pytest.raises(ModelLoadError, match="no está instalado"):
            LLMServer(model_file)


def test_llm_server_raises_if_model_missing(tmp_path: Path):
    missing = tmp_path / "no_existe.gguf"
    with patch("src.engine.llm_server.Llama", MagicMock()):
        with pytest.raises(ModelLoadError, match="No se encontró"):
            LLMServer(missing)


def test_llm_server_wraps_load_error_when_cpu_only(model_file: Path):
    mock_llama = MagicMock(side_effect=RuntimeError("archivo corrupto"))
    with patch("src.engine.llm_server.Llama", mock_llama):
        with pytest.raises(ModelLoadError, match="archivo corrupto"):
            LLMServer(model_file, n_gpu_layers=0)
    mock_llama.assert_called_once()


def test_llm_server_falls_back_to_cpu_when_gpu_load_fails(model_file: Path):
    mock_llama = MagicMock(side_effect=[RuntimeError("VRAM insuficiente"), MagicMock()])
    with patch("src.engine.llm_server.Llama", mock_llama):
        server = LLMServer(model_file, n_gpu_layers=-1)

    assert server.n_gpu_layers == 0
    assert mock_llama.call_count == 2
    assert mock_llama.call_args_list[0].kwargs["n_gpu_layers"] == -1
    assert mock_llama.call_args_list[1].kwargs["n_gpu_layers"] == 0


def test_llm_server_raises_if_gpu_and_cpu_both_fail(model_file: Path):
    mock_llama = MagicMock(side_effect=RuntimeError("sin memoria"))
    with patch("src.engine.llm_server.Llama", mock_llama):
        with pytest.raises(ModelLoadError, match="tanto en GPU como en CPU"):
            LLMServer(model_file, n_gpu_layers=-1)
    assert mock_llama.call_count == 2


# ---------------------------------------------------------------------------
# LLMServer - generación síncrona
# ---------------------------------------------------------------------------


def _build_server(model_file: Path, llm_instance: MagicMock) -> LLMServer:
    mock_llama_cls = MagicMock(return_value=llm_instance)
    with patch("src.engine.llm_server.Llama", mock_llama_cls):
        return LLMServer(model_file, n_gpu_layers=0)


def test_generate_returns_text(model_file: Path):
    llm_instance = MagicMock(return_value=make_completion("Hola mundo"))
    server = _build_server(model_file, llm_instance)

    result = server.generate("Saluda")

    assert result == "Hola mundo"
    llm_instance.assert_called_once()
    assert llm_instance.call_args.kwargs["stream"] is False


def test_generate_raises_on_empty_prompt(model_file: Path):
    server = _build_server(model_file, MagicMock())
    with pytest.raises(ValueError):
        server.generate("")


def test_generate_wraps_backend_errors(model_file: Path):
    llm_instance = MagicMock(side_effect=RuntimeError("contexto agotado"))
    server = _build_server(model_file, llm_instance)

    with pytest.raises(GenerationError, match="contexto agotado"):
        server.generate("hola")


def test_generate_raises_on_malformed_response(model_file: Path):
    llm_instance = MagicMock(return_value={"choices": []})
    server = _build_server(model_file, llm_instance)

    with pytest.raises(GenerationError, match="formato inesperado"):
        server.generate("hola")


def test_generate_uses_custom_config(model_file: Path):
    llm_instance = MagicMock(return_value=make_completion("ok"))
    server = _build_server(model_file, llm_instance)
    config = GenerationConfig(max_tokens=16, temperature=0.1, stop=["\n"])

    server.generate("hola", config=config)

    kwargs = llm_instance.call_args.kwargs
    assert kwargs["max_tokens"] == 16
    assert kwargs["temperature"] == 0.1
    assert kwargs["stop"] == ["\n"]


# ---------------------------------------------------------------------------
# LLMServer - generación por streaming
# ---------------------------------------------------------------------------


def test_generate_stream_yields_chunks(model_file: Path):
    chunks = [make_completion("Hola"), make_completion(" mundo", finish_reason=None)]
    llm_instance = MagicMock(return_value=iter(chunks))
    server = _build_server(model_file, llm_instance)

    result = list(server.generate_stream("Saluda"))

    assert result == ["Hola", " mundo"]
    assert llm_instance.call_args.kwargs["stream"] is True


def test_generate_stream_raises_on_empty_prompt(model_file: Path):
    server = _build_server(model_file, MagicMock())
    with pytest.raises(ValueError):
        list(server.generate_stream(""))


def test_generate_stream_wraps_setup_errors(model_file: Path):
    llm_instance = MagicMock(side_effect=RuntimeError("fallo de setup"))
    server = _build_server(model_file, llm_instance)

    with pytest.raises(GenerationError, match="fallo de setup"):
        list(server.generate_stream("hola"))


def test_generate_stream_raises_on_malformed_chunk(model_file: Path):
    llm_instance = MagicMock(return_value=iter([{"choices": []}]))
    server = _build_server(model_file, llm_instance)

    with pytest.raises(GenerationError, match="formato inesperado"):
        list(server.generate_stream("hola"))


def test_generate_stream_skips_empty_text_chunks(model_file: Path):
    chunks = [make_completion(""), make_completion("fin")]
    llm_instance = MagicMock(return_value=iter(chunks))
    server = _build_server(model_file, llm_instance)

    assert list(server.generate_stream("hola")) == ["fin"]


# ---------------------------------------------------------------------------
# LLMServer - cierre / context manager
# ---------------------------------------------------------------------------


def test_close_calls_underlying_close(model_file: Path):
    llm_instance = MagicMock()
    server = _build_server(model_file, llm_instance)

    server.close()

    llm_instance.close.assert_called_once()


def test_context_manager_closes_on_exit(model_file: Path):
    llm_instance = MagicMock()
    mock_llama_cls = MagicMock(return_value=llm_instance)

    with patch("src.engine.llm_server.Llama", mock_llama_cls):
        with LLMServer(model_file, n_gpu_layers=0) as server:
            assert isinstance(server, LLMServer)

    llm_instance.close.assert_called_once()


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------


def test_run_benchmark_measures_ttft_and_tokens_per_second(model_file: Path):
    def fake_stream(prompt, config=None):
        time.sleep(0.01)
        yield "Hola"
        yield " mundo"

    llm_instance = MagicMock()
    server = _build_server(model_file, llm_instance)
    server.generate_stream = fake_stream  # type: ignore[method-assign]

    with patch("src.engine.benchmarks.current_ram_mb", return_value=123.0), patch(
        "src.engine.benchmarks.current_vram_mb", return_value=456.0
    ):
        result = run_benchmark(server, "hola")

    assert result.tokens_generated == 2
    assert result.ttft_seconds > 0
    assert result.total_seconds >= result.ttft_seconds
    assert result.tokens_per_second > 0
    assert result.ram_mb == 123.0
    assert result.vram_mb == 456.0


def test_run_benchmark_zero_tokens_has_zero_throughput(model_file: Path):
    llm_instance = MagicMock()
    server = _build_server(model_file, llm_instance)
    server.generate_stream = lambda prompt, config=None: iter([])  # type: ignore[method-assign]

    result = run_benchmark(server, "hola")

    assert result.tokens_generated == 0
    assert result.tokens_per_second == 0.0


def test_current_ram_mb_returns_none_without_psutil():
    with patch("src.engine.benchmarks.psutil", None):
        assert current_ram_mb() is None


def test_current_vram_mb_returns_none_without_nvidia_smi():
    with patch("src.engine.benchmarks.shutil.which", return_value=None):
        assert current_vram_mb() is None


def test_current_vram_mb_parses_nvidia_smi_output():
    fake_result = MagicMock(returncode=0, stdout="1024\n")
    with patch("src.engine.benchmarks.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
        "src.engine.benchmarks.subprocess.run", return_value=fake_result
    ):
        assert current_vram_mb() == 1024.0


def _result(ttft: float, total: float, tokens: int) -> BenchmarkResult:
    return BenchmarkResult(
        ttft_seconds=ttft,
        total_seconds=total,
        tokens_generated=tokens,
        tokens_per_second=tokens / total if total else 0.0,
        ram_mb=None,
        vram_mb=None,
    )


def test_decode_tokens_per_second_excludes_ttft():
    # 11 tokens: el primero llega en el TTFT, los 10 restantes en 2 s de decodificación.
    assert decode_tokens_per_second(_result(ttft=3.0, total=5.0, tokens=11)) == pytest.approx(5.0)


def test_decode_tokens_per_second_is_zero_without_decode_phase():
    assert decode_tokens_per_second(_result(ttft=1.0, total=1.0, tokens=1)) == 0.0


def test_build_benchmark_prompts_are_distinct():
    prompts = build_benchmark_prompts(15)
    assert len(prompts) == 15
    assert len(set(prompts)) == 15


def test_run_suite_resets_cache_only_for_cold_runs(model_file: Path):
    llm_instance = MagicMock()
    server = _build_server(model_file, llm_instance)
    events = []
    llm_instance.reset.side_effect = lambda: events.append("reset")

    def fake_stream(prompt, config=None):
        events.append("run")
        yield "a"
        yield "b"

    server.generate_stream = fake_stream  # type: ignore[method-assign]

    suite = run_suite(server, GenerationConfig(), runs=2, cached_repeats=2, warmup=1)

    # calentamiento + 2 sin caché + 1 que carga el caché: cada una precedida por reset;
    # las 2 del control con caché, no.
    assert events == ["reset", "run"] * 4 + ["run", "run"]
    assert len(suite["cold_runs"]) == 2
    assert len(suite["cached_runs"]) == 2
    assert suite["summary"]["tokens_generated"]["median"] == 2


def test_run_suite_rejects_zero_runs(model_file: Path):
    server = _build_server(model_file, MagicMock())
    with pytest.raises(ValueError):
        run_suite(server, GenerationConfig(), runs=0)


def test_benchmarks_main_writes_json_report(model_file: Path, tmp_path: Path):
    llm_instance = MagicMock()
    llm_instance.n_threads = 4
    server = _build_server(model_file, llm_instance)
    server.generate_stream = lambda prompt, config=None: iter(["a", "b", "c"])  # type: ignore[method-assign]
    output = tmp_path / "bench.json"

    with patch("src.engine.benchmarks.LLMServer", return_value=server):
        exit_code = benchmarks.main(
            ["--model-path", str(model_file), "--runs", "2", "--cached-repeats", "0", "--output", str(output)]
        )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["protocol"]["runs"] == 2
    assert len(report["cold_runs"]) == 2
    assert report["cached_runs"] == []
    assert report["cached_summary"] is None
    assert report["summary"]["tokens_generated"]["median"] == 3
