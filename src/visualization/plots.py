"""Gráficos padronizados para a avaliação de classificadores."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve


def save_figure(figure: plt.Figure, filename: str, output_dir: str | Path = "images") -> Path:
    """Salva uma figura em PNG e retorna seu caminho."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    figure.savefig(path, dpi=300, bbox_inches="tight")
    return path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Matriz de confusão") -> plt.Figure:
    """Cria a matriz de confusão usada na avaliação final."""

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=axis, colorbar=False, normalize="true")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, auc_score: float, title: str = "Curva ROC") -> plt.Figure:
    """Cria a curva ROC com a referência de classificador aleatório."""

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.4f}")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Aleatório")
    axis.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos", title=title)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, title: str = "Curva Precision-Recall") -> plt.Figure:
    """Cria a curva precision-recall e sua linha de prevalência."""

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, label="Modelo")
    axis.axhline(np.mean(y_true), linestyle="--", color="gray", label="Prevalência")
    axis.set(xlabel="Recall", ylabel="Precision", title=title)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def plot_ks_curve(y_true: np.ndarray, y_proba: np.ndarray, title: str = "Curva KS") -> plt.Figure:
    """Exibe TPR, FPR e a maior distância KS."""

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    ks_values = tpr - fpr
    best_index = int(np.argmax(ks_values))

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(thresholds, tpr, label="TPR (recall)")
    axis.plot(thresholds, fpr, label="FPR")
    axis.scatter(thresholds[best_index], ks_values[best_index], color="crimson", label=f"KS = {ks_values[best_index]:.4f}")
    axis.set(xlabel="Threshold", ylabel="Taxa", title=title)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure
