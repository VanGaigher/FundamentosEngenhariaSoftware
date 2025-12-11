"""
Módulo para treinamento de modelos.
"""

from json import dump
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ks_2samp

PROCESSED_DATA_FOLDER = r"C:\Users\vanes\Documents\02-Estudos\FundamentosEngenhariaSoftware\data\processed"
ARTIFACTS_FOLDER = r"C:\Users\vanes\Documents\02-Estudos\FundamentosEngenhariaSoftware\artifacts"

X_TRAIN_PATH = os.path.join(PROCESSED_DATA_FOLDER, "X_train_imputed.csv")
Y_TRAIN_PATH = os.path.join(PROCESSED_DATA_FOLDER, "y_train.csv").squeeze()
X_TEST_PATH = os.path.join(PROCESSED_DATA_FOLDER, "X_test_imputed.csv")
Y_TEST_PATH = os.path.join(PROCESSED_DATA_FOLDER, "y_test.csv").squeeze()


def train_model():
    """
    Função para treinar o modelo.
    """
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)

    top_features = [
        "entrega_doc_2_nan",
        "score_10",
        "grupo_categorias",
        "entrega_doc_1",
        "score_6",
        "score_9",
        "score_1",
        "score_7",
        "valor_compra",
        "score_4",
        "score_2",
        "entrega_doc_3",
        "score_3",
    ]

    best_params = {
        "criterion": "entropy",
        "max_depth": 6,
        "min_samples_leaf": 20,
        "min_samples_split": 20,
    }

    X_train_top = X_train[top_features]

    final_model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        criterion=best_params["criterion"],
    )

    final_model.fit(X_train_top, y_train)

    dump(final_model, os.path.join(ARTIFACTS_FOLDER, "modelo_final.joblib"))
    print("Modelo treinado e salvo com sucesso!")
