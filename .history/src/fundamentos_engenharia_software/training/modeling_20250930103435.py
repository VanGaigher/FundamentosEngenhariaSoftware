"""
Módulo para treinamento de modelos.
"""

import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ks_2samp

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


def select_features_important():
    """
    Função para selecionar as features mais importantes.
    """
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

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
    X_train_selected = X_train.copy()
    X_test_selected = X_test.copy()

    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]

    return X_train_selected, X_test_selected


def train_model(best_params, X_train, y_train):
    """
    Função para treinar o modelo.
    """

    final_dt_model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        criterion=best_params["criterion"],
    )

    final_dt_model.fit(X_train, y_train)

    return final_dt_model


def make_predictions(final_dt_model, X_train, X_test):
    """
    Função para fazer previsões com o modelo treinado.
    """

    y_train_proba_final = final_dt_model.predict_proba(X_train)[:, 1]
    y_pred_proba_final = final_dt_model.predict_proba(X_test)[:, 1]

    y_pred_final = final_dt_model.predict(X_test)

    return y_train_proba_final, y_pred_proba_final, y_pred_final


def model_evaluating(
    final_dt_model,
    y_train_proba_final,
    y_pred_proba_final,
    y_train,
    y_test,
    y_pred_final,
):
    """
    Função para avaliar o modelo.
    """

    train_auc = roc_auc_score(y_train, y_train_proba_final)
    test_auc = roc_auc_score(y_test, y_pred_proba_final)
    precision = precision_score(y_test, y_pred_final)
    recall = recall_score(y_test, y_pred_final)

    ks_stat, ks_p = ks_2samp(
        y_pred_proba_final[y_test == 1], y_pred_proba_final[y_test == 0]
    )

    return {
        "AUC Treino": train_auc,
        "AUC Teste": test_auc,
        "KS Statistic": ks_stat,
        "KS P-value": ks_p,
        "Precision": precision,
        "Recall": recall,
    }
