"""
Módulo de "features engineering para pipeline de detecção de fraudes.

Neste módulo, vamos ter funções para criar e transformar features.
As etapas aqui colocadas são baseadas em EDA experimental feita previamente em notebook Jupyter.

As funções incluem:
- create_cumulative_fraud_percentage: Criação de coluna de percentage acumulada de fraude por categoria de produto.
- extract_least_frequent_categories: Extração de categorias menos frequentes.
- create_other_category_values: Agrupamento de países menos frequentes.
- create_document_delivery_indicators: Criação de colunas de indicadores de entrega de documentos.
- encode_categorical_columns: Encoding de colunas categóricas.
- main_feature_engineering: Função principal para orquestrar a criação e transformação de features.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.fundamentos_engenharia_software.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)


class CategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, percentage_cutoff: int = 80, top_n: int = 685):
        self.percentage_cutoff = percentage_cutoff
        self.top_n = top_n

        self.least_frequent_categories = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CategoryGrouper:
        try:
            df_copy = pd.concat([X, y], axis=1)

            item_cat = df_copy.categoria_produto.value_counts().reset_index()
            item_cat.columns = ["categoria_produto", "qnt_registros"]

            fraude_cat = (
                df_copy.groupby("categoria_produto")["fraude"]
                .sum()
                .reset_index()
            )

            df_item_fraude = pd.merge(
                item_cat, fraude_cat, on="categoria_produto", how="left"
            )
            df_item_fraude = df_item_fraude.sort_values(
                by="fraude", ascending=False
            ).reset_index(drop=True)

            df_item_fraude["percent_cumsum_fraude"] = (
                df_item_fraude["fraude"].cumsum()
                / df_item_fraude["fraude"].sum()
                * 100
            )

            df_item_fraude["reaches_cutoff"] = (
                df_item_fraude["percent_cumsum_fraude"]
                <= self.percentage_cutoff
            )

            produtos_categorias = df_item_fraude[self.top_n :]

            self.least_frequent_categories = (
                produtos_categorias.categoria_produto.to_list()
            )

        except KeyError as e:
            print(f"Erro no CategoryGrouper: Coluna {e} não encontrada.")
            raise

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X_copy = X.copy()

        X_copy["grupo_categorias"] = X_copy["categoria_produto"]

        X_copy.loc[
            X_copy["grupo_categorias"].isin(self.least_frequent_categories),
            "grupo_categorias",
        ] = "categorias_outros"

        return X_copy


class CountryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, countries_to_keep: List[str]):
        self.countries_to_keep = countries_to_keep

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> CountryGrouper:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X_copy = X.copy()
        X_copy["paises_agrupados"] = np.where(
            X_copy["pais"].isin(self.countries_to_keep),
            X_copy["pais"],
            "Outros",
        )

        return X_copy


class DocumentFeatureCreator(BaseEstimator, TransformerMixin):

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> DocumentFeatureCreator:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        df_copy = X.copy()

        df_copy["entrega_doc_2_nan"] = np.where(
            df_copy["entrega_doc_2"].isnull(), 1, 0
        )
        df_copy["entrega_doc"] = (
            df_copy[["entrega_doc_1", "entrega_doc_2", "entrega_doc_3"]]
            .any(axis=1)
            .astype(int)
        )

        return df_copy


class ColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders_ = {}
        self.columns_ = []

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> ColumnEncoder:
        self.columns_ = List(X.select_dtypes(include="object").columns)

        for col in self.columns_:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders_[col] = le
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()

        for col in self.columns_:
            if col in self.encoders_:
                le = self.encoders_[col]
                X_copy[col] = le.transform(X_copy[col].astype(str))
        return X_copy


class FeatureEngineer:

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.feature_pipeline = self._get_pipeline()

    def _get_pipeline(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("category_grouper", CategoryGrouper()),
                (
                    "country_grouper",
                    CountryGrouper(countries_to_keep=["BR", "AR"]),
                ),
                ("document_feature_creator", DocumentFeatureCreator()),
                ("column_encoder", ColumnEncoder()),
            ]
        )

    def read_raw_data(self) -> None:
        try:
            print("Lendo os dados brutos...")
            self.data = pd.read_excel(self.input_path)
            print("Dados brutos lidos com sucesso.")
        except FileNotFoundError:
            print(
                f"Arquivo não encontrado em {self.input_path}. Verifique o caminho."
            )
            raise Exception

    def run_feature_engineer(self) -> None:
        self.read_raw_data()

        y = self.data["fraude"]
        X = self.data.drop(columns=["fraude"])

        data_processed = self.feature_pipeline.fit_transform(X, y)
        data_processed["fraude"] = y.values

        self.processed_data = data_processed

        self.save_feature_engineer_results()

    def save_feature_engineer_results(self) -> None:
        try:
            print("Salvando os dados processados...")
            self.processed_data.to_csv(self.output_path, index=False)
            print(f"Dados processados salvos em {self.output_path}.")
        except PermissionError as e:
            print(f'("Permissão negada ao salvar os dados processados.") {e}')
            raise Exception


def create_features_and_encode() -> None:
    feature_engineer = FeatureEngineer(
        input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH
    )
    feature_engineer.run_feature_engineer()
