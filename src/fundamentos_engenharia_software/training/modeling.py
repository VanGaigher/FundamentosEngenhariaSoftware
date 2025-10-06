"""
Módulo de treinamento do modelo.

Este script é responsável por treinar o modelo de classificação.
Ele carrega os dados de treino pré-processados, treina um modelo
de Árvore de Decisão com hiperparâmetros definidos e salva o artefato
treinado para uso posterior na etapa de avaliação.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
from src.fundamentos_engenharia_software.config import (
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    TOP_FEATURES,
    MODEL_PATH,
)


def train_model() -> None:
    """
    Treina o modelo de árvore de decisão e o salva em disco.

    Esta função executa os seguintes passos:
    1. Carrega os conjuntos de dados de treino (features e target).
    2. Seleciona as features mais importantes (definidas em TOP_FEATURES).
    3. Instancia um modelo de Árvore de Decisão com hiperparâmetros pré-definidos.
    4. Treina o modelo com os dados de treino.
    5. Salva o objeto do modelo treinado no caminho especificado em MODEL_PATH.
    """
    try:
        print("Iniciando o treinamento do modelo.")

        X_train = pd.read_csv(X_TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

        best_params = {
            "criterion": "entropy",
            "max_depth": 6,
            "min_samples_leaf": 20,
            "min_samples_split": 20,
        }

        X_train_top = X_train[TOP_FEATURES]

        final_model = DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            criterion=best_params["criterion"],
        )

        final_model.fit(X_train_top, y_train)

        dump(final_model, MODEL_PATH)
        print("Modelo treinado e salvo com sucesso!")
    except FileNotFoundError as e:
        print(f"Arquivo de dados de treino não encontrado: {e}")
        raise
    except Exception as e:
        print(f"Ocorreu um erro durante o treinamento: {e}")
        raise
