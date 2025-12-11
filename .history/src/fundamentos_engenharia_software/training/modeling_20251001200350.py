"""
Módulo para treinamento de modelos.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
from fundamentos_engenharia_software.config import (
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    TOP_FEATURES,
    MODEL_PATH,
)


def train_model():
    """
    Função para treinar o modelo.
    """
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
