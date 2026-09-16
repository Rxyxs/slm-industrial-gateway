"""Paquete de evaluación de fidelidad y alucinación para respuestas del SLM."""

from .evaluator import FaithfulnessEvaluator, FaithfulnessResult

__all__ = ["FaithfulnessEvaluator", "FaithfulnessResult"]
