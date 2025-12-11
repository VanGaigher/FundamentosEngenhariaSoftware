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

def select_features_important(X_train, X_test):
    """
    Função para selecionar as features mais importantes.
    """
    top_features = ['entrega_doc_2_nan', 
                    'score_10', 
                    'grupo_categorias', 
                    'entrega_doc_1', 
                    'score_6', 
                    'score_9', 
                    'score_1', 
                    'score_7', 
                    'valor_compra', 
                    'score_4', 
                    'score_2', 
                    'entrega_doc_3',
                     'score_3']
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    return X_train_selected, X_test_selected
6               score_7
12         valor_compra
3               score_4
1               score_2
11        entrega_doc_3
2               score_3
Name: feature, dtype: object
    X_train_selected = X_train[top_features]
    X_test_selected = X_test[top_features]
    return X_train_selected, X_test_selected


def train_model(best_params):
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

    final_dt_model.fit(X_train, y_train)
    
    return final_dt_model

def evaluate_model(final_dt_model):
    """
    Função para avaliar o modelo.
    """
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    y_pred = final_dt_model.predict(X_test)
    
    return y_pred



# Treina no conjunto under-sampled
final_dt_model.fit(X_train, y_train)

# Fazer previsões (Probabilidades)
y_train_proba_final = final_dt_model.predict_proba(X_train)[:, 1]
y_pred_proba_final = final_dt_model.predict_proba(X_test)[:, 1]

# Classes preditas
y_pred_final = final_dt_model.predict(X_test)
