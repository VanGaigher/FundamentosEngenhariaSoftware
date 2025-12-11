import pandas as pd

from preprocessing.feature_engineering import CountryGrouper


def test_country_grouper_transform():
    """
    Testa o nosso trasformador CountryGde países
    """

    # Arrange
    data = {
        "country": ["BR", "AR", "US", "FR", "DE", "BR", "IT", "AR", "ES"],
        "value": [10, 20, 30, 40, 50, 60, 70, 80, 90],
    }
    sample_df = pd.DataFrame(data)
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
