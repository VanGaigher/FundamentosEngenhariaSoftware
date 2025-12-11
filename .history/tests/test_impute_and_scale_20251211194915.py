import pandas as pd
import numpy as np
import pytest

from src.fundamentos_engenharia_software.preprocessing.impute_and_scale import (
    MissingImputerScaler,
    DataScalerAndImputer,
)


# definir o que quero testar das classes acima
@pytest.fixture
def numeric_missing_dataframe() -> pd.DataFrame:
    """
    DataFrame de exemplo para testar o MissingImputerScaler.

    - Contém colunas numéricas com valores ausentes.
    - Contém uma coluna extra que não será transformada.
    """
    data = {
        "idade": [25, 30, np.nan, 40, 35, np.nan, 50],
        "salario": [5000, 6000, 5500, np.nan, 6500, 7000, np.nan],
        "bonus": [
            200,
            300,
            250,
            400,
            350,
            300,
            450,
        ],  # coluna que não terá NaN
        "departamento": [
            "Vendas",
            "TI",
            "TI",
            "RH",
            "Vendas",
            "TI",
            "RH",
        ],  # não numérica
    }
    return pd.DataFrame(data)
