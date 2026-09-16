"""Pruebas del evaluador de fidelidad y alucinación (src/evaluation)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import FaithfulnessEvaluator


def _mock_metric(score: float, passed: bool, reason: str = "ok") -> MagicMock:
    metric = MagicMock()
    metric.score = score
    metric.reason = reason
    metric.is_successful.return_value = passed
    metric.measure.return_value = score
    return metric


@patch("src.evaluation.evaluator.HallucinationMetric")
@patch("src.evaluation.evaluator.FaithfulnessMetric")
def test_evaluate_faithful_response(
    mock_faithfulness_cls: MagicMock, mock_hallucination_cls: MagicMock
) -> None:
    mock_faithfulness_cls.return_value = _mock_metric(0.95, True, "Bien sustentada por el contexto")
    mock_hallucination_cls.return_value = _mock_metric(0.05, True)

    evaluator = FaithfulnessEvaluator(judge_model="mock-model")
    result = evaluator.evaluate(
        input_query="¿Cuál es la capital de Francia?",
        actual_output="La capital de Francia es París.",
        retrieval_context=["París es la capital de Francia."],
    )

    assert result.faithfulness_score == 0.95
    assert result.faithfulness_passed is True
    assert result.hallucination_score == 0.05
    assert result.hallucination_passed is True
    assert result.reason == "Bien sustentada por el contexto"
    assert result.passed is True


@patch("src.evaluation.evaluator.HallucinationMetric")
@patch("src.evaluation.evaluator.FaithfulnessMetric")
def test_evaluate_hallucinated_response(
    mock_faithfulness_cls: MagicMock, mock_hallucination_cls: MagicMock
) -> None:
    mock_faithfulness_cls.return_value = _mock_metric(
        0.2, False, "No sustentada por el contexto"
    )
    mock_hallucination_cls.return_value = _mock_metric(0.8, False)

    evaluator = FaithfulnessEvaluator(judge_model="mock-model")
    result = evaluator.evaluate(
        input_query="¿Cuál es la capital de Francia?",
        actual_output="La capital de Francia es Roma.",
        retrieval_context=["París es la capital de Francia."],
    )

    assert result.faithfulness_passed is False
    assert result.hallucination_passed is False
    assert result.passed is False


@patch("src.evaluation.evaluator.HallucinationMetric")
@patch("src.evaluation.evaluator.FaithfulnessMetric")
def test_evaluator_configures_metrics_with_judge_model_and_thresholds(
    mock_faithfulness_cls: MagicMock, mock_hallucination_cls: MagicMock
) -> None:
    mock_faithfulness_cls.return_value = _mock_metric(1.0, True)
    mock_hallucination_cls.return_value = _mock_metric(0.0, True)

    FaithfulnessEvaluator(
        judge_model="local-judge",
        faithfulness_threshold=0.8,
        hallucination_threshold=0.2,
    )

    _, faithfulness_kwargs = mock_faithfulness_cls.call_args
    assert faithfulness_kwargs["model"] == "local-judge"
    assert faithfulness_kwargs["threshold"] == 0.8

    _, hallucination_kwargs = mock_hallucination_cls.call_args
    assert hallucination_kwargs["model"] == "local-judge"
    assert hallucination_kwargs["threshold"] == 0.2


@patch("src.evaluation.evaluator.HallucinationMetric")
@patch("src.evaluation.evaluator.FaithfulnessMetric")
def test_evaluate_passes_retrieval_context_to_test_case(
    mock_faithfulness_cls: MagicMock, mock_hallucination_cls: MagicMock
) -> None:
    faithfulness_metric = _mock_metric(0.9, True)
    hallucination_metric = _mock_metric(0.1, True)
    mock_faithfulness_cls.return_value = faithfulness_metric
    mock_hallucination_cls.return_value = hallucination_metric

    evaluator = FaithfulnessEvaluator(judge_model="mock-model")
    context = ["Dato A.", "Dato B."]
    evaluator.evaluate(
        input_query="pregunta",
        actual_output="respuesta",
        retrieval_context=context,
    )

    faithfulness_test_case = faithfulness_metric.measure.call_args.args[0]
    assert faithfulness_test_case.retrieval_context == context
    assert faithfulness_test_case.context == context
    assert faithfulness_test_case.input == "pregunta"
    assert faithfulness_test_case.actual_output == "respuesta"
