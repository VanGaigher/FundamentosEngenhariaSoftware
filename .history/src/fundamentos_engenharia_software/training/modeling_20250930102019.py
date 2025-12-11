"""
Módulo para treinamento de modelos.
"""

import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

PROCESSED_DATA_FOLDER = r"C:\Users\vanes\Documents\02-Estudos\FundamentosEngenhariaSoftware\data\processed"

X_TRAIN_PATH = os.path.join(
    PROCESSED_DATA_FOLDER, "X_train_imputed_scaled.csv"
)
Y_TRAIN_PATH = os.path.join(PROCESSED_DATA_FOLDER, "y_train.csv")
X_TEST_PATH = os.path.join(PROCESSED_DATA_FOLDER, "X_test_imputed_scaled.csv")
Y_TEST_PATH = os.path.join(PROCESSED_DATA_FOLDER, "y_test.csv")

best_params = {
    "criterion": "entropy",
    "max_depth": 6,
    "min_samples_leaf": 20,
    "min_samples_split": 20,
}


def train_model():
    """
    Função para treinar o modelo.
    """
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)

    final_dt_model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        criterion=best_params["criterion"],
    )


# Treina no conjunto under-sampled
final_dt_model.fit(X_train, y_train)

# Fazer previsões (Probabilidades)
y_train_proba_final = final_dt_model.predict_proba(X_train)[:, 1]
y_pred_proba_final = final_dt_model.predict_proba(X_test)[:, 1]

# Classes preditas
y_pred_final = final_dt_model.predict(X_test)
