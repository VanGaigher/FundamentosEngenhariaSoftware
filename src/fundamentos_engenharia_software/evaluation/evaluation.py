"""
Módulo para avaliação de modelos.

"""

import joblib
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.fundamentos_engenharia_software.config import (
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    TOP_FEATURES,
    MODEL_PATH,
)


def load_data_and_artifacts():
    """
    Função para carregar os dados e artefatos necessários.
    """

    model = joblib.load(MODEL_PATH)
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    return model, X_train, y_train, X_test, y_test


def make_predictions(X_train, X_test, model):
    """
    Função para fazer previsões com o modelo carregado.
    """

    y_train_proba = model.predict_proba(X_train[TOP_FEATURES])[:, 1]
    y_pred_proba = model.predict_proba(X_test[TOP_FEATURES])[:, 1]
    y_pred = model.predict(X_test[TOP_FEATURES])

    return y_train_proba, y_pred_proba, y_pred


def calculate_metrics(y_train, y_test, y_train_proba, y_pred_proba, y_pred):
    """
    Função para calcular métricas de avaliação do modelo.
    """

    # Métricas
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    ks_stat, ks_p = ks_2samp(
        y_pred_proba[y_test == 1], y_pred_proba[y_test == 0]
    )

    print(f"AUC Treino: {train_auc:.4f}")
    print(f"AUC Teste: {test_auc:.4f}")
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"KS P-value: {ks_p:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


def evaluate_model():
    """
    Função principal para avaliar o modelo.
    """
    model, X_train, y_train, X_test, y_test = load_data_and_artifacts()
    y_train_proba, y_pred_proba, y_pred = make_predictions(
        X_train, X_test, model
    )
    calculate_metrics(y_train, y_test, y_train_proba, y_pred_proba, y_pred)
