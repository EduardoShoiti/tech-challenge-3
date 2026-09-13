"""Avaliação, seleção de threshold e interpretabilidade dos modelos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from src.visualization import plot_confusion_matrix, plot_ks_curve, plot_precision_recall_curve, plot_roc_curve, save_figure


def find_threshold(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "ks",
    min_precision: float = 0.60,
) -> float:
    """Escolhe threshold por KS máximo ou por maior recall com precision mínima."""

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    if strategy == "ks":
        return float(thresholds[int(np.argmax(tpr - fpr))])

    if strategy == "recall":
        candidates = np.unique(y_proba)
        valid: list[tuple[float, float]] = []
        for threshold in candidates:
            prediction = (y_proba >= threshold).astype(int)
            precision = precision_score(y_true, prediction, zero_division=0)
            if precision >= min_precision:
                valid.append((recall_score(y_true, prediction, zero_division=0), float(threshold)))
        if not valid:
            raise ValueError("Nenhum threshold atende à precision mínima definida.")
        return max(valid, key=lambda item: item[0])[1]

    raise ValueError("strategy deve ser 'ks' ou 'recall'.")


def metrics_at_threshold(
    y_true: pd.Series | np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Calcula métricas de classificação preservando as probabilidades para AUC."""

    prediction = (y_proba >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return {
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "F1-Score": f1_score(y_true, prediction, zero_division=0),
        "Accuracy": accuracy_score(y_true, prediction),
        "AUC": roc_auc_score(y_true, y_proba),
        "KS": float(np.max(tpr - fpr)),
        "Average Precision": average_precision_score(y_true, y_proba),
        "Prevalência": float(np.mean(y_true)),
    }


def permutation_importance_table(
    model: Any, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """Retorna importância por permutação no nível das colunas originais."""

    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    return (
        pd.DataFrame(
            {"Variável": X.columns, "Importância média": result.importances_mean, "Desvio padrão": result.importances_std}
        )
        .sort_values("Importância média", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_model(
    model: Any,
    datasets: dict[str, tuple[pd.DataFrame, pd.Series]],
    threshold: str | float = "ks",
    threshold_dataset: str = "Validação",
    min_precision: float = 0.60,
    save_images: bool = False,
    output_dir: str | Path = "images",
    image_suffix: str = "model",
) -> tuple[pd.DataFrame, float]:
    """Avalia um modelo em um ou mais conjuntos sem recalibrar no teste.

    Se ``threshold`` for texto, ele é aprendido apenas em
    ``threshold_dataset``. Para avaliar o teste, informe o valor numérico já
    escolhido na validação.
    """

    if isinstance(threshold, str):
        if threshold_dataset not in datasets:
            raise ValueError("O dataset de validação para seleção de threshold não foi informado.")
        X_reference, y_reference = datasets[threshold_dataset]
        selected_threshold = find_threshold(
            y_reference,
            model.predict_proba(X_reference)[:, 1],
            strategy=threshold,
            min_precision=min_precision,
        )
    else:
        selected_threshold = float(threshold)

    rows: list[dict[str, float | str]] = []
    for name, (X_data, y_data) in datasets.items():
        probabilities = model.predict_proba(X_data)[:, 1]
        row: dict[str, float | str] = {"Dataset": name}
        row.update(metrics_at_threshold(y_data, probabilities, selected_threshold))
        rows.append(row)

        if save_images:
            prediction = (probabilities >= selected_threshold).astype(int)
            auc_score = float(row["AUC"])
            figures = {
                "matriz_confusao": plot_confusion_matrix(y_data, prediction, f"Matriz de confusão — {name}"),
                "curva_roc": plot_roc_curve(y_data, probabilities, auc_score, f"Curva ROC — {name}"),
                "precision_recall": plot_precision_recall_curve(y_data, probabilities, f"Precision-Recall — {name}"),
                "curvas_ks": plot_ks_curve(y_data, probabilities, f"Curva KS — {name}"),
            }
            for chart_name, figure in figures.items():
                save_figure(figure, f"{chart_name}_{image_suffix}_{name}.png", output_dir)

    return pd.DataFrame(rows).set_index("Dataset"), selected_threshold
