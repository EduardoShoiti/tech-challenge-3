"""Fábricas e rotinas de validação dos modelos avaliados no projeto."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def make_cv(random_state: int = 42, n_splits: int = 5) -> StratifiedKFold:
    """Retorna a validação cruzada estratificada usada no treinamento."""

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def score_model(model: Pipeline, X_train: Any, y_train: Any, cv: StratifiedKFold) -> float:
    """Calcula a ROC AUC média de CV somente sobre o conjunto de treino."""

    scores = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    return float(scores.mean())


def create_baseline_models(preprocessor: Any, random_state: int = 42) -> dict[str, Pipeline]:
    """Cria os cinco baselines comparados no notebook de modelagem."""

    return {
        "Logistic Regression": Pipeline(
            [("preprocessing", preprocessor), ("model", LogisticRegression(max_iter=2000, random_state=random_state))]
        ),
        "Decision Tree": Pipeline(
            [("preprocessing", preprocessor), ("model", DecisionTreeClassifier(random_state=random_state))]
        ),
        "Random Forest": Pipeline(
            [("preprocessing", preprocessor), ("model", RandomForestClassifier(random_state=random_state, n_jobs=1))]
        ),
        "LightGBM": Pipeline(
            [("preprocessing", preprocessor), ("model", LGBMClassifier(random_state=random_state, verbosity=-1, n_jobs=1))]
        ),
        "Gaussian Naive Bayes": Pipeline(
            [("preprocessing", preprocessor), ("model", GaussianNB())]
        ),
    }


def create_tuned_models(
    preprocessor: Any,
    best_params: dict[str, dict[str, Any]],
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Monta os modelos finais a partir dos parâmetros encontrados pelo Optuna.

    ``best_params`` deve usar as chaves: ``lr``, ``dt``, ``rf``, ``lgbm`` e
    ``nb``, correspondentes aos estudos do notebook.
    """

    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessing", preprocessor),
                ("model", LogisticRegression(**best_params["lr"], solver="liblinear", max_iter=2000, random_state=random_state)),
            ]
        ),
        "Decision Tree": Pipeline(
            [("preprocessing", preprocessor), ("model", DecisionTreeClassifier(**best_params["dt"], random_state=random_state))]
        ),
        "Random Forest": Pipeline(
            [("preprocessing", preprocessor), ("model", RandomForestClassifier(**best_params["rf"], random_state=random_state, n_jobs=-1))]
        ),
        "LightGBM": Pipeline(
            [
                ("preprocessing", preprocessor),
                ("model", LGBMClassifier(**best_params["lgbm"], subsample_freq=1, random_state=random_state, verbosity=-1, n_jobs=-1)),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            [("preprocessing", preprocessor), ("model", GaussianNB(**best_params["nb"]))]
        ),
    }
