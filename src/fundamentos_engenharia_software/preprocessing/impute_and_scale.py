"""
Módulo para limpeza de dados.
"""

import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.fundamentos_engenharia_software.config import (
    PROCESSED_DATA_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    COLS_TO_USE,
)


def read_and_split_data():
    """
    Função para ler os dados e fazer o split entre treino e teste.
    """
    data_with_features = pd.read_csv(PROCESSED_DATA_PATH)

    X = data_with_features[COLS_TO_USE].drop(columns="fraude")
    y = data_with_features["fraude"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def extract_missing_columns(X_train):
    """
    Função para extrair as colunas numéricas com valores ausentes (NaN) no conjunto de treino.
    """

    # Selecionar apenas colunas numéricas que possuem valores ausentes (NaN)
    cols_com_missing = X_train.select_dtypes(include="number").columns[
        X_train.select_dtypes(include="number").isna().sum() > 0
    ]

    return cols_com_missing


def scale_missing_columns(X_train, X_test, missing_columns):
    """
    Função para escalar as colunas numéricas usando Min-Max Scaling.
    """
    X_train_scaled = X_train[missing_columns].copy()
    X_test_scaled = X_test[missing_columns].copy()

    scaler = MinMaxScaler()

    X_train_scaled[missing_columns] = scaler.fit_transform(
        X_train[missing_columns]
    )

    X_test_scaled[missing_columns] = scaler.transform(X_test[missing_columns])

    return X_train_scaled, X_test_scaled


def impute_missing_data(
    X_train, X_test, X_train_scaled, X_test_scaled, missing_columns
):
    """
    Função para imputar valores ausentes usando KNN Imputer.
    """
    imputer = KNNImputer(n_neighbors=5)

    X_train_num = X_train_scaled.copy()

    X_train_imputed = imputer.fit_transform(X_train_num)

    X_train_imputed = pd.DataFrame(
        X_train_imputed, columns=X_train_num.columns, index=X_train_num.index
    )

    X_test_num = X_test_scaled.copy()
    X_test_imputed = imputer.transform(X_test_num)
    X_test_imputed = pd.DataFrame(
        X_test_imputed, columns=X_test_num.columns, index=X_test_num.index
    )

    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()

    X_train_copy[missing_columns] = X_train_imputed
    X_test_copy[missing_columns] = X_test_imputed

    return X_train_copy, X_test_copy


def impute_and_scale():
    """
    Função para imputar valores ausentes e escalar os dados.
    """
    X_train, X_test, y_train, y_test = read_and_split_data()

    missing_columns = extract_missing_columns(X_train)
    X_train_scaled, X_test_scaled = scale_missing_columns(
        X_train, X_test, missing_columns
    )

    X_train_imputed, X_test_imputed = impute_missing_data(
        X_train, X_test, X_train_scaled, X_test_scaled, missing_columns
    )

    X_train_imputed.to_csv(os.path.join(X_TRAIN_PATH), index=False)
    X_test_imputed.to_csv(os.path.join(X_TEST_PATH), index=False)
    y_train.to_csv(os.path.join(Y_TRAIN_PATH), index=False)
    y_test.to_csv(os.path.join(Y_TEST_PATH), index=False)
