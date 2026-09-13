"""Transformadores usados na pipeline de alfabetização.

As funções deste módulo extraem a lógica utilizada no notebook de modelagem.
O pré-processamento é sempre ajustado dentro de um ``sklearn.Pipeline`` para
evitar que estatísticas de validação ou teste sejam usadas no treinamento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Codifica categorias pela frequência observada no conjunto de treino.

    Categorias inéditas e valores ausentes na transformação recebem frequência
    zero. Isso permite tratar variáveis categóricas com mais categorias sem
    expandir a matriz de atributos como no one-hot encoding.
    """

    def __init__(self) -> None:
        self.frequency_maps_: dict[str, dict[object, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FrequencyEncoder":
        frame = pd.DataFrame(X).copy()
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.frequency_maps_ = {
            column: frame[column].value_counts(normalize=True, dropna=False).to_dict()
            for column in frame.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        for column in self.feature_names_in_:
            frame[column] = frame[column].map(self.frequency_maps_[column]).fillna(0.0)
        return frame

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        features = self.feature_names_in_ if input_features is None else input_features
        return np.asarray(features, dtype=object)


def create_preprocessor(
    onehot_cols: list[str] | None = None,
    frequency_cols: list[str] | None = None,
    standard_cols: list[str] | None = None,
    remainder: str = "passthrough",
) -> ColumnTransformer:
    """Cria o pré-processador do projeto.

    A imputação fica dentro das subpipelines para que medianas e categorias de
    preenchimento sejam aprendidas exclusivamente no treino de cada fold.
    """

    transformers: list[tuple[str, object, list[str]]] = []

    if onehot_cols:
        transformers.append(
            (
                "onehot",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                onehot_cols,
            )
        )

    if frequency_cols:
        transformers.append(("frequency", FrequencyEncoder(), frequency_cols))

    if standard_cols:
        transformers.append(
            (
                "standard",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                standard_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder=remainder,
        verbose_feature_names_out=False,
    )
