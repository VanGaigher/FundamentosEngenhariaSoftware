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
from typing import List, Optional
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

from src.fundamentos_engenharia_software.config import (
    PROCESSED_DATA_PATH,
    PROCESSED_DATA_FOLDER,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    COLS_TO_USE,
)


class MissingImputerScaler(BaseEstimator, TransformerMixin):
    """
    Classe para escalar e imputar valores ausentes em colunas numéricas.
    """

    def __init__(
        self,
        missing_columns: List[str],
        n_neighbors: int = 5,
    ):
        self.missing_columns = missing_columns
        self.n_neighbors = n_neighbors

        self.scaler_ = None  # o underscore indica que é um atributo interno a ser aprendido
        self.imputer_ = None  # o underscore indica que é um atributo interno a ser aprendido

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Ajusta o escalonador e o imputador aos dados de treino.
        """
        self.scaler_ = MinMaxScaler()
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)

        scaled_data = self.scaler_.fit_transform(X[self.missing_columns])
        self.imputer_.fit(scaled_data)

        return self

    def transform(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transforma os dados escalonando e imputando valores ausentes.
        """
        X_copy = X.copy()

        transformed_data = self.scaler_.transform(X_copy[self.missing_columns])
        transformed_data = self.imputer_.transform(transformed_data)

        X_copy[self.missing_columns] = pd.DataFrame(
            transformed_data,
            columns=self.missing_columns,
            index=X_copy.index,
        )
        return X_copy


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

    def run(self) -> None:
        """
        Função para imputar valores ausentes e escalar os dados.
        """
        self._read_and_split_data()
        self._extract_missing_columns()
        self.missing_transformer = MissingImputerScaler(
            self.missing_columns, n_neighbors=self.n_neighbors
        )

        self.X_train_imputed = self.missing_transformer.fit_transform(
            self.X_train
        )
        self.X_test_imputed = self.missing_transformer.transform(self.X_test)

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
