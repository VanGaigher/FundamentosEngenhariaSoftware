import pandas as pd
import numpy as np
import pytest

from fundamentos_engenharia_software.preprocessing.feature_engineering import (
    CountryGrouper,
    ColumnEncoder,
)


# Arrange
@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Cria um DataFrame de exemplo para testes."""
    sample_df = {
        "country": ["BR", "AR", "US", "FR", "DE", "BR", "IT", "AR", "ES"],
        "value": [10, 20, 30, 40, 50, 60, 70, 80, 90],
    }
    return pd.DataFrame(sample_df)


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
        result["paises_agrupados"], expected_values, check_names=False
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

    assert len(encoder.encoders_["pais"].classes_) == 3
