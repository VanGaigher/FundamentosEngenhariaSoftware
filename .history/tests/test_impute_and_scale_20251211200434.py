import pandas as pd
import numpy as np
import pytest
from io import StringIO
from src.fundamentos_engenharia_software.preprocessing.impute_and_scale import (
    DataScalerAndImputer,
)


# definir o que quero testar das classes acima
@pytest.fixture
def scaler_imputer_dataframe() -> pd.DataFrame:
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
cols_to_use = ["idade", "salario", "bonus", "departamento"]
def test_missing_imputer_scaler_fit(tmp_path, scaler_imputer_dataframe):
    #Salvar o DataFrame de teste em um arquivo CSV temporário
    csv_path = tmp_path / "test_data.csv"
    scaler_imputer_dataframe.to_csv(csv_path, index=False)

    # Criar a instância do DataScalerAndImputer
    preprocessor = DataScalerAndImputer(input_path=str(csv_path),
                                        output_x_train_path=str(tmp_path / "X_train.csv"),
                                        output_y_train_path=str(tmp_path / "y_train.csv"),
                                        output_x_test_path=str(tmp_path / "X_test.csv"),
                                        output_y_test_path=str(tmp_path / "y_test.csv"),
                                        artifacts_path=str(tmp_path),
                                        test_size=0.3,
                                        random_state=42,
                                        n_neighbors=2
                                        cols_to_use=cols_to_use
    )
    #rodar pipeline
    preprocessor.run()

    # verificar colunas com missing
    assert set(preprocessor.missing_columns)=={"idade", "salario"}

    # verificar se as colunas imputadas não tem nan
    assert preprocessor.X_train_imputed[preprocessor.missing_columns].isnull().sum().sum() == 0
    assert preprocessor.X_test_imputed[preprocessor.missing_columns].isnull().sum().sum() == 0

    # verificar se colunas categóricas não foram alteradas
    pd.testing.assert_series_equal(
        preprocessor.X_train_imputed["departamento"],
        preprocessor.X_train["departamento"],
        check_dtype=False,
    )

    #verificar se alvo foi preservado
    assert set(preprocessor.y_train.unique())<= {0, 1}
    assert set(preprocessor.y_test.unique())<= {0, 1}

