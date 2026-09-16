"""Evaluación de fidelidad (Faithfulness) y tasa de alucinación de respuestas del SLM.

Usa DeepEval para comparar la respuesta generada por el SLM contra el contexto
recuperado (RAG), detectando afirmaciones no sustentadas por ese contexto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from deepeval.metrics import FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
DEFAULT_FAITHFULNESS_THRESHOLD = 0.7
DEFAULT_HALLUCINATION_THRESHOLD = 0.3


@dataclass
class FaithfulnessResult:
    """Resultado combinado de fidelidad y alucinación para una respuesta del SLM."""

    faithfulness_score: float
    faithfulness_passed: bool
    hallucination_score: float
    hallucination_passed: bool
    reason: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.faithfulness_passed and self.hallucination_passed


class FaithfulnessEvaluator:
    """Evalúa si la respuesta del SLM está sustentada en el contexto recuperado.

    El modelo usado como juez es configurable (`judge_model`) para permitir
    apuntar a un endpoint local u OpenAI-compatible en despliegues aislados de
    red, en vez de depender de una API externa.
    """

    def __init__(
        self,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        faithfulness_threshold: float = DEFAULT_FAITHFULNESS_THRESHOLD,
        hallucination_threshold: float = DEFAULT_HALLUCINATION_THRESHOLD,
    ) -> None:
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=faithfulness_threshold,
            model=judge_model,
            include_reason=True,
        )
        self.hallucination_metric = HallucinationMetric(
            threshold=hallucination_threshold,
            model=judge_model,
            include_reason=True,
        )

    def evaluate(
        self,
        input_query: str,
        actual_output: str,
        retrieval_context: list[str],
    ) -> FaithfulnessResult:
        """Mide fidelidad y alucinación de `actual_output` frente a `retrieval_context`."""
        test_case = LLMTestCase(
            input=input_query,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
            context=retrieval_context,
        )

        self.faithfulness_metric.measure(test_case)
        self.hallucination_metric.measure(test_case)

        return FaithfulnessResult(
            faithfulness_score=self.faithfulness_metric.score,
            faithfulness_passed=bool(self.faithfulness_metric.is_successful()),
            hallucination_score=self.hallucination_metric.score,
            hallucination_passed=bool(self.hallucination_metric.is_successful()),
            reason=self.faithfulness_metric.reason,
        )
