import pandas as pd
import numpy as np
import pytest

from src.fundamentos_engenharia_software.preprocessing.feature_engineering import (
    CountryGrouper,
    ColumnEncoder,
    CategoryGrouper,
)


# Arrange
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Cria um DataFrame de exemplo para testes."""
    sample_df = {
        "pais": ["BR", "AR", "US", "FR", "DE", "BR", "IT", "AR", "ES"],
        "valor": [10, 20, 30, 40, 50, 60, 70, 80, 90],
    }
    return pd.DataFrame(sample_df)


@pytest.fixture
def category_dataframe() -> pd.DataFrame:
    """Cria um Dataframe de exemplo para testar o CategoryGrouper."""
    data = {
        "categoria_produto": ["A", "B", "A", "C", "B", "A", "D", "C", "B"],
        "fraude": [1, 1, 0, 1, 0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


# TESTES


def test_country_grouper_transform(sample_df):
    """
    Testa o nosso trasformador CountryGde países
    """
    # Arrange
    countries_to_keep = ["BR", "AR"]
    grouper = CountryGrouper(countries_to_keep=countries_to_keep)

    # Act
    result = grouper.transform(sample_df)

    # Assert
    expected_values = [
        "BR",
        "AR",
        "Outros",
        "Outros",
        "Outros",
        "BR",
        "Outros",
        "AR",
        "Outros",
    ]
    assert "paises_agrupados" in result.columns

    pd.testing.assert_series_equal(
        result["paises_agrupados"],
        pd.Series(expected_values),
        check_names=False,
    )

    assert result.shape == (9, 3)


def test_column_encoder_fit_transform(sample_df):
    """
    Testa o nosso trasformador ColumnEncoder
    """

    # Arrange
    encoder = ColumnEncoder()

    # Act
    result = encoder.fit_transform(sample_df)

    # Assert
    assert np.issubdtype(result["pais"].dtype, np.integer)

    pd.testing.assert_series_equal(result["valor"], sample_df["valor"])

    assert "pais" in encoder.encoders_
    assert "valor" not in encoder.encoders_

    assert len(encoder.encoders_["pais"].classes_) == 7


# "pais": ["BR", "AR", "US", "FR", "DE", "BR", "IT", "AR", "ES"]
@pytest.mark.parametrize(
    "countries_to_keep, expect_other_count",
    [
        (["BR", "AR"], 5),
        (["BR"], 7),
        ([], 9),
        (["US", "FR", "ES"], 6),
    ],
)
def test_country_grouper_parametrized(
    sample_df, countries_to_keep, expect_other_count
):
    """
    Testa o CountryGrouper com diferentes parâmetros.
    """
    grouper = CountryGrouper(countries_to_keep=countries_to_keep)
    result = grouper.transform(sample_df)

    assert (
        result["paises_agrupados"].value_counts().get("Outros", 0)
        == expect_other_count
    )


def test_category_grouper_fit(category_dataframe):
    """
    Testa se o CategoryGrouper aprende as categorias corretas no fit e as transforma corretamente .
    """
    X = category_dataframe.drop(columns="fraude")
    y = category_dataframe["fraude"]

    # Act
    grouper = CategoryGrouper(top_n=2)

    grouper.fit(X, y)

    # # Assert
    # "categoria_produto": ["A", "B", "A", "C", "B", "A", "D", "C", "B"]
    # expected_values = [
    #     "A",
    #     "B",
    #     "A",
    #     "categorias_outros",
    #     "B",
    #     "A",
    #     "categorias_outros",
    #     "categorias_outros",
    #     "B",
    # ]

    assert isinstance(grouper.least_frequent_categories, list)
    assert "B" in grouper.least_frequent_categories
    assert "D" in grouper.least_frequent_categories
    assert len(grouper.least_frequent_categories) == 2


def test_category_grouper_transform(category_dataframe):
    """
    Testa se o CategoryGrouper transforma corretamente as categorias.
    """
    X = category_dataframe.drop(columns="fraude")
    y = category_dataframe["fraude"]

    grouper = CategoryGrouper(top_n=2)

    grouper.fit(X, y)
    result_df = grouper.transform(X)

    assert "grupo_categorias" in result_df.columns

    assert (
        result_df["grupo_categorias"].isin(["categorias_outros"]).sum()
        == len(category_dataframe) - 6
    )
