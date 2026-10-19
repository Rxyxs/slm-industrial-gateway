"""Tests del harness de evaluación offline (`src/evaluation/run_offline_eval.py`):
lógica de clasificación de resultados, cálculo de Safety Block Rate y del
intervalo de Wilson -- con el SLM mockeado, sin pesos GGUF ni red.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evaluation.run_offline_eval import (
    PromptOutcome,
    _summarize,
    score_answer,
    wilson_ci,
)


def test_wilson_ci_is_narrower_with_more_samples():
    lo_small, hi_small = wilson_ci(9, 10)
    lo_big, hi_big = wilson_ci(90, 100)

    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_wilson_ci_handles_zero_n():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_bounds_stay_within_unit_interval():
    lo, hi = wilson_ci(100, 100)
    assert 0.0 <= lo <= hi <= 1.0


def test_summarize_counts_blocked_over_limit_prompt_as_correct():
    prompts = [{"id": "safety_00", "category": "safety_boundary", "expected_over_limit": True}]
    outcomes = [PromptOutcome("safety_00", "safety_boundary", "blocked_safety", None, True)]

    summary = _summarize(prompts, outcomes, elapsed_seconds=1.0)

    assert summary["safety_block_rate"]["correct"] == 1
    assert summary["safety_block_rate"]["n"] == 1
    assert summary["safety_block_rate"]["value"] == 1.0


def test_summarize_counts_unblocked_over_limit_prompt_as_incorrect():
    outcomes = [PromptOutcome("safety_00", "safety_boundary", "ok", "sensor_anomaly_check", True)]

    summary = _summarize([], outcomes, elapsed_seconds=1.0)

    assert summary["safety_block_rate"]["correct"] == 0


def test_summarize_counts_allowed_within_limit_prompt_as_correct():
    """Control negativo: un prompt dentro de rango que NO se bloquea es el comportamiento correcto."""
    outcomes = [PromptOutcome("safety_01", "safety_boundary", "ok", None, False)]

    summary = _summarize([], outcomes, elapsed_seconds=1.0)

    assert summary["safety_block_rate"]["correct"] == 1


def test_summarize_counts_blocked_within_limit_prompt_as_incorrect():
    """Bloquear algo que estaba dentro de rango también es un error (falso positivo)."""
    outcomes = [PromptOutcome("safety_01", "safety_boundary", "blocked_safety", None, False)]

    summary = _summarize([], outcomes, elapsed_seconds=1.0)

    assert summary["safety_block_rate"]["correct"] == 0


def test_summarize_ignores_non_safety_categories_in_block_rate():
    outcomes = [
        PromptOutcome("rul_00", "rul", "ok", "calculate_rul", None),
        PromptOutcome("safety_00", "safety_boundary", "blocked_safety", None, True),
    ]

    summary = _summarize([], outcomes, elapsed_seconds=1.0)

    assert summary["safety_block_rate"]["n"] == 1


def test_summarize_reports_faithfulness_and_relevancy_as_unmeasured():
    summary = _summarize([], [], elapsed_seconds=0.0)

    assert summary["faithfulness"] is None
    assert summary["answer_relevancy"] is None
    assert "OPENAI_API_KEY" in summary["unmeasured_note"]


def test_summarize_is_json_serializable():
    outcomes = [PromptOutcome("anomaly_00", "anomaly_check", "ok", "sensor_anomaly_check", None)]

    summary = _summarize([], outcomes, elapsed_seconds=2.5)

    json.dumps(summary)  # no debe lanzar


def test_summarize_counts_categories_and_statuses():
    outcomes = [
        PromptOutcome("anomaly_00", "anomaly_check", "ok", "sensor_anomaly_check", None),
        PromptOutcome("anomaly_01", "anomaly_check", "error", None, None),
        PromptOutcome("rul_00", "rul", "ok", "calculate_rul", None),
    ]

    summary = _summarize([], outcomes, elapsed_seconds=1.0)

    assert summary["categories"]["anomaly_check"] == 2
    assert summary["categories"]["rul"] == 1
    assert summary["status_by_category"]["anomaly_check"]["ok"] == 1
    assert summary["status_by_category"]["anomaly_check"]["error"] == 1


def test_score_answer_anomaly_matches_tool_flag():
    item = {"category": "anomaly_check", "expected_anomalous": True}
    assert score_answer(item, "sensor_anomaly_check", {"is_last_reading_anomalous": True})
    assert not score_answer(item, "sensor_anomaly_check", {"is_last_reading_anomalous": False})
    assert not score_answer(item, "calculate_rul", {"is_last_reading_anomalous": True})


def test_score_answer_rul_tolerance_and_no_trend():
    item = {"category": "rul", "expected_rul_hours": 20.0}
    assert score_answer(item, "calculate_rul", {"rul_estimate": 20.9})
    assert not score_answer(item, "calculate_rul", {"rul_estimate": 22.0})
    flat = {"category": "rul", "expected_rul_hours": None}
    assert score_answer(flat, "calculate_rul", {"rul_estimate": None})
    assert not score_answer(flat, "calculate_rul", {"rul_estimate": 5.0})


def test_score_answer_duckdb_finds_expected_value_in_rows():
    item = {"category": "duckdb_analytics", "expected_value": 476.6512}
    assert score_answer(item, "query_duckdb", {"rows": [[476.65]]})
    assert not score_answer(item, "query_duckdb", {"rows": [[400.0]]})
    assert not score_answer(item, "query_duckdb", None)


def test_eval_prompts_are_real_plant_cases_with_expected_answers():
    prompts = json.loads((Path(__file__).resolve().parents[1] / "data" / "eval_prompts.json").read_text(encoding="utf-8"))
    assert len(prompts) == 100
    by_cat = {}
    for p in prompts:
        by_cat.setdefault(p["category"], []).append(p)
    assert {k: len(v) for k, v in by_cat.items()} == {
        "anomaly_check": 25, "rul": 25, "safety_boundary": 25, "duckdb_analytics": 25}
    assert sum(p["expected_anomalous"] for p in by_cat["anomaly_check"]) == 12
    assert sum(p["expected_over_limit"] for p in by_cat["safety_boundary"]) == 13
    assert all("flotation_hourly" in p["prompt"] for p in by_cat["duckdb_analytics"])
