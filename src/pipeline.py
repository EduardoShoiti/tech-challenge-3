"""Orquestrador reproduzível para o modelo de alfabetização.

Execução a partir da raiz do repositório:

    python -m src.pipeline

O fluxo reproduz a estratégia definida no notebook de modelagem: treino e
validação em 2023, teste temporal em 2024, remoção de atributos com risco de
data leakage e avaliação final com threshold calibrado apenas na validação.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation import evaluate_model, permutation_importance_table
from src.modeling import create_tuned_models
from src.preprocessing import create_preprocessor


RANDOM_STATE = 42
TARGET = "alfabetizado"
TIME_COLUMN = "ano"

# Colunas que não estariam disponíveis antes da avaliação ou que não
# generalizam para novos alunos/territórios.
ID_COLUMNS = ["id_aluno", "id_municipio", "codigo_uf", "rede"]
LEAKAGE_COLUMNS = [
    "proficiencia",
    "peso_aluno",
    "presenca",
    "preenchimento_caderno",
    "caderno",
    "municipio_taxa_alfabetizacao_pct",
    "municipio_media_portugues",
    "uf_taxa_alfabetizacao_pct",
    "uf_media_portugues",
    "brasil_taxa_alfabetizacao_pct",
]

# Parâmetros vencedores encontrados no Optuna no notebook de modelagem.
TUNED_PARAMS = {
    "lr": {"C": 0.005920890155973574, "penalty": "l1", "class_weight": "balanced"},
    "dt": {
        "criterion": "entropy",
        "max_depth": 18,
        "min_samples_split": 22,
        "min_samples_leaf": 17,
        "ccp_alpha": 4.054085695276339e-05,
        "class_weight": "balanced",
    },
    "rf": {
        "max_depth": 12,
        "min_samples_split": 22,
        "min_samples_leaf": 3,
        "max_features": "log2",
        "class_weight": None,
    },
    "lgbm": {
        "max_depth": 10,
        "learning_rate": 0.057489180054056295,
        "num_leaves": 39,
        "min_child_samples": 11,
        "subsample": 0.8316874291291989,
        "colsample_bytree": 0.8060188414980882,
        "reg_alpha": 0.13340666414920144,
        "reg_lambda": 0.002874514553045364,
        "class_weight": "balanced",
    },
    "nb": {"var_smoothing": 1.8522071756196817e-12},
}

MODEL_NAMES = {
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
    "gaussian_nb": "Gaussian Naive Bayes",
}


def columns_to_remove(frame: pd.DataFrame) -> list[str]:
    """Determina remoções usando somente informações de treino de 2023."""

    fixed_columns = {TARGET, TIME_COLUMN, *ID_COLUMNS, *LEAKAGE_COLUMNS}
    fixed_columns.update(column for column in frame.columns if column.startswith("meta_"))
    constants = {column for column in frame.columns if frame[column].nunique(dropna=False) <= 1}
    return sorted((fixed_columns | constants) & set(frame.columns))


def split_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Cria treino/validação em 2023 e teste temporal em 2024."""

    train_year = frame.loc[frame[TIME_COLUMN] == 2023].copy()
    test_year = frame.loc[frame[TIME_COLUMN] == 2024].copy()
    if train_year.empty or test_year.empty:
        raise ValueError("A base precisa conter registros dos anos de 2023 e 2024.")

    train_frame, validation_frame = train_test_split(
        train_year,
        test_size=0.20,
        stratify=train_year[TARGET],
        random_state=RANDOM_STATE,
    )
    features = [column for column in frame.columns if column not in columns_to_remove(train_year)]
    return train_frame, validation_frame, test_year, features


def build_preprocessor() -> object:
    """Define as transformações escolhidas e usadas no notebook."""

    return create_preprocessor(
        onehot_cols=["regiao", "rede_desc", "escolaridade"],
        frequency_cols=["id_escola", "sigla_uf", "caracterizacao_orgao_gestor", "escolaridade_formacao"],
        standard_cols=["pib", "populacao"],
    )


def run(
    data_path: Path,
    output_dir: Path,
    model_key: str = "random_forest",
    save_images: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa treinamento, avaliação temporal e geração de artefatos.

    Retorna as tabelas de métricas de desenvolvimento e de teste. Também salva
    métricas e importância por permutação em ``output_dir``.
    """

    if model_key not in MODEL_NAMES:
        raise ValueError(f"Modelo inválido: {model_key}. Opções: {', '.join(MODEL_NAMES)}")
    if not data_path.exists():
        raise FileNotFoundError(f"Base de modelagem não encontrada: {data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(data_path)
    train_frame, validation_frame, test_frame, features = split_data(frame)

    models = create_tuned_models(build_preprocessor(), TUNED_PARAMS, random_state=RANDOM_STATE)
    model_name = MODEL_NAMES[model_key]
    model = models[model_name]

    X_train, y_train = train_frame[features], train_frame[TARGET]
    X_validation, y_validation = validation_frame[features], validation_frame[TARGET]
    X_test, y_test = test_frame[features], test_frame[TARGET]

    print(f"Treinando {model_name} com {len(features)} atributos...")
    model.fit(X_train, y_train)

    development_metrics, threshold = evaluate_model(
        model=model,
        datasets={"Treino": (X_train, y_train), "Validação": (X_validation, y_validation)},
        threshold="recall",
        threshold_dataset="Validação",
        min_precision=0.60,
        save_images=save_images,
        output_dir=output_dir / "figures",
        image_suffix=model_key,
    )
    test_metrics, _ = evaluate_model(
        model=model,
        datasets={"Teste": (X_test, y_test)},
        threshold=threshold,
        save_images=save_images,
        output_dir=output_dir / "figures",
        image_suffix=model_key,
    )

    importance = permutation_importance_table(model, X_test, y_test)
    development_metrics.to_csv(output_dir / "metrics_development.csv")
    test_metrics.to_csv(output_dir / "metrics_test.csv")
    importance.to_csv(output_dir / "permutation_importance_test.csv", index=False)

    print(f"Threshold selecionado na validação: {threshold:.4f}")
    print("\nMétricas de teste:")
    print(test_metrics.round(4).to_string())
    print(f"\nArtefatos salvos em: {output_dir.resolve()}")
    return development_metrics, test_metrics


def parse_args() -> argparse.Namespace:
    """Lê opções de execução pela linha de comando."""

    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Treina e avalia o modelo de alfabetização.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=project_dir / "data" / "sample" / "data_sample_modeling.parquet",
        help="Caminho da base parquet enriquecida.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "reports" / "modeling",
        help="Diretório em que métricas e gráficos serão gravados.",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="random_forest")
    parser.add_argument("--no-images", action="store_true", help="Não gera gráficos de avaliação.")
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do módulo executável."""

    args = parse_args()
    run(args.data_path, args.output_dir, args.model, save_images=not args.no_images)


if __name__ == "__main__":
    main()
