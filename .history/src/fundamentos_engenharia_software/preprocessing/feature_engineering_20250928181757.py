"""
Módulo para engenharia de features.
"""

import pandas as pd
import numpy as np

RAW_DATA_PATH = (
    r"C:\Users\vanes\Documents\02-Estudos\FundamentosEngenhariaSoftware\data\dados.xlsx"
)
PROCESSED_DATA_PATH = r"C:\Users\vanes\Documents\02-Estudos\FundamentosEngenhariaSoftware\data\dados_com_features.csv"


def create_cumulative_fraud_percentage(df):
    """
    Função para criar a coluna de percentage acumulada de fraude por categoria de produto.
    """
    df_copy = df.copy()

    item_cat = df_copy.categoria_produto.value_counts().reset_index()
    item_cat.columns = ["categoria_produto", "qnt_registros"]

    fraude_cat = df_copy.groupby("categoria_produto")["fraude"].sum().reset_index()

    df_item_fraude = pd.merge(item_cat, fraude_cat, on="categoria_produto", how="left")
    df_item_fraude = df_item_fraude.sort_values(
        by="fraude", ascending=False
    ).reset_index(drop=True)

    df_item_fraude["percent_cumsum_fraude"] = (
        df_item_fraude["fraude"].cumsum() / df_copy["fraude"].sum() * 100
    )

    return df_item_fraude


def extract_least_frequent_categories(df, percentage_cutoff=80, top_n=685):
    """
    Função para extrair a lista de categorias menos frequentes.
    """
    df_copy = df.copy()

    df_copy["reaches_80"] = df_copy["percent_cumsum_fraude"] <= percentage_cutoff

    produtos_categorias = df_copy[685:]

    lista_categorias_outros = produtos_categorias.categoria_produto.to_list()

    return lista_categorias_outros


def create_other_category_values(df, lista_categoria_outros):
    """
    Função para criar os valores da categoria 'outros'.
    """
    df_copy = df.copy()

    df_copy["grupo_categorias"] = df_copy["categoria_produto"]

    df_copy.loc[
        df_copy["grupo_categorias"].isin(lista_categoria_outros), "grupo_categorias"
    ] = "categorias_outros"

    return df_copy


def group_countries(df, countries_to_keep=["BR", "AR"]):
    """
    Função para agrupar países em uma categoria 'Outros'.
    """
    df_copy = df.copy()
    df_copy["paises_agrupados"] = np.where(
        df_copy["pais"].isin(countries_to_keep), df_copy["pais"], "Outros"
    )

    return df_copy


def create_document_columns(df):
    """
    Função para criar colunas de indicadores de entrega de documentos.
    """
    df_copy = df.copy()

    df_copy["entrega_doc_2_nan"] = np.where(df_copy["entrega_doc_2"].isnull(), 1, 0)
    df_copy["entrega_doc"] = (
        df_copy[["entrega_doc_1", "entrega_doc_2", "entrega_doc_3"]]
        .any(axis=1)
        .astype(int)
    )

    return df_copy


def create_features():
    """
    Função para criar novas features a partir dos dados brutos.
    """
    df = pd.read_excel(RAW_DATA_PATH)

    # Calcular a percentage acumulada
    df_with_percentage_column = create_cumulative_fraud_percentage(df)

    # Extrair lista de categorias a serem mantidas
    other_categories_list = extract_least_frequent_categories(
        df_with_percentage_column, percentage_cutoff=80, top_n=685
    )

    # Fazer agrupamento das categorias no dataframe original
    df_with_other_categories = create_other_category_values(df, other_categories_list)

    # fazer agrupamento dos países menos frequentes
    df_with_contries_grouped = group_countries(
        df_with_other_categories, countries_to_keep=["BR", "AR"]
    )

    # criar coluna com indicador de entrega de documentos
    df_with_doc_columns = create_document_columns(df_with_contries_grouped)

    df_with_doc_columns.to_csv(PROCESSED_DATA_PATH, index=False)
