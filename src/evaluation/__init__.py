"""Métricas e rotinas de avaliação de modelos."""

from .metrics import evaluate_model, find_threshold, metrics_at_threshold, permutation_importance_table

__all__ = ["evaluate_model", "find_threshold", "metrics_at_threshold", "permutation_importance_table"]
