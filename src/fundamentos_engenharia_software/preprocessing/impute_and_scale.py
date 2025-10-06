"""
Módulo para limpeza de dados.
As etapas aqui colocadas são baseadas em EDA experimental feita previamente em notebook Jupyter.
As funções incluem:
- read_and_split_data: Leitura dos dados e split entre treino e teste.
- extract_missing_columns: Extração das colunas numéricas com valores ausentes (NaN).
- scale_missing_columns: Escalonamento das colunas numéricas usando Min-Max Scaling.
- impute_missing_data: Imputação de valores ausentes usando KNN Imputer.
- impute_and_scale: Função principal para orquestrar a limpeza dos dados.

"""

import os
from typing import List
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.fundamentos_engenharia_software.config import (
    PROCESSED_DATA_PATH,
    PROCESSED_DATA_FOLDER,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    COLS_TO_USE,
)


class DataScalerAndImputer:
    """
    Classe para escalonar e imputar dados.
    """

    def __init__(
        self,
        input_path: str,
        output_x_train_path: str,
        output_x_test_path: str,
        output_y_train_path: str,
        output_y_test_path: str,
        cols_to_use: List[str],
        test_size: float = 0.3,
        random_state: int = 42,
        n_neighbors: int = 5,
    ):
        self.input_path = input_path
        self.output_x_train_path = output_x_train_path
        self.output_x_test_path = output_x_test_path
        self.output_y_train_path = output_y_train_path
        self.output_y_test_path = output_y_test_path
        self.cols_to_use = cols_to_use
        self.test_size = test_size
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def _read_and_split_data(
        self,
    ) -> None:
        """
        Função para ler os dados e fazer o split entre treino e teste.
        Args:
            None
        """
        try:
            print("Lendo dados de:", self.input_path)
            data_with_features = pd.read_csv(self.input_path)

            X = data_with_features[self.cols_to_use].drop(columns="fraude")
            y = data_with_features["fraude"]

            self.X_train, self.X_test, self.y_train, self.y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            )
            return True
        except FileNotFoundError as e:
            print(f"Arquivo não encontrado: {e}")
            raise

    def _extract_missing_columns(self) -> pd.Index:
        """
        Função para extrair as colunas numéricas com valores ausentes (NaN) no conjunto de treino.
        Args:
            X_train (pd.DataFrame): Conjunto de treino com features.
        """

        # Selecionar apenas colunas numéricas que possuem valores ausentes (NaN)
        cols_com_missing = self.X_train.select_dtypes(
            include="number"
        ).columns[
            self.X_train.select_dtypes(include="number").isna().sum() > 0
        ]

        self.missing_columns = cols_com_missing

    def _scale_missing_columns(self):
        """
        Função para escalar as colunas numéricas usando Min-Max Scaling.

        """
        X_train_scaled = self.X_train[self.missing_columns].copy()
        X_test_scaled = self.X_test[self.missing_columns].copy()

        scaler = MinMaxScaler()

        X_train_scaled[self.missing_columns] = scaler.fit_transform(
            self.X_train[self.missing_columns]
        )

        X_test_scaled[self.missing_columns] = scaler.transform(
            self.X_test[self.missing_columns]
        )

        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled

    def _impute_missing_data(self):
        """
        Função para imputar valores ausentes usando KNN Imputer.
        """
        imputer = KNNImputer(n_neighbors=self.n_neighbors)

        X_train_num = self.X_train_scaled.copy()

        X_train_imputed = imputer.fit_transform(X_train_num)

        X_train_imputed = pd.DataFrame(
            X_train_imputed,
            columns=X_train_num.columns,
            index=X_train_num.index,
        )

        X_test_num = self.X_test_scaled.copy()
        X_test_imputed = imputer.transform(X_test_num)
        X_test_imputed = pd.DataFrame(
            X_test_imputed, columns=X_test_num.columns, index=X_test_num.index
        )

        X_train_copy = self.X_train.copy()
        X_test_copy = self.X_test.copy()

        X_train_copy[self.missing_columns] = X_train_imputed
        X_test_copy[self.missing_columns] = X_test_imputed

        self.X_train_imputed = X_train_copy
        self.X_test_imputed = X_test_copy

    def run(self) -> None:
        """
        Função para imputar valores ausentes e escalar os dados.
        """
        self._read_and_split_data()
        self._extract_missing_columns()
        self._scale_missing_columns()
        self._impute_missing_data()

        try:
            print(
                f"Salvando dados processados em: {os.path.dirname(PROCESSED_DATA_FOLDER)}"
            )
            self.X_train_imputed.to_csv(self.output_x_train_path, index=False)
            self.X_test_imputed.to_csv(self.output_x_test_path, index=False)
            self.y_train.to_csv(self.output_y_train_path, index=False)
            self.y_test.to_csv(self.output_y_test_path, index=False)
            print("Dados processados salvos com sucesso.")
        except Exception as e:
            print(f"Erro ao salvar os dados processados: {e}")
            raise


def impute_and_scale():
    scaler_and_imputer = DataScalerAndImputer(
        input_path=PROCESSED_DATA_PATH,
        output_x_train_path=X_TRAIN_PATH,
        output_x_test_path=X_TEST_PATH,
        output_y_train_path=Y_TRAIN_PATH,
        output_y_test_path=Y_TEST_PATH,
        cols_to_use=COLS_TO_USE,
    )

    scaler_and_imputer.run()
