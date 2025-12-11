import pandas as pd
import numpy as np
import pytest

from src.fundamentos_engenharia_software.preprocessing.impute_and_scale import (
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
        "bonus": [200, 300, 250, 400, 350, 300, 450],  # sem NaN
        "departamento": [
            "Vendas",
            "TI",
            "TI",
            "RH",
            "Vendas",
            "TI",
            "RH",
        ],  # não numérica
        "fraude": [0, 1, 0, 1, 0, 1, 0],  # alvo
    }
    return pd.DataFrame(data)


# Act: testar se ele aprende os parametros corretamente
def test_missing_imputer_scaler_fit(data):
    scaler = DataScalerAndImputer()
