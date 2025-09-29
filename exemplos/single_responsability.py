import pandas as pd
from sklearn.model_selection import train_test_split

# EXEMPLO DE VIOLAÇÃO DO PRINCÍPIO DA RESPONSABILIDADE ÚNICA

# Aqui temos a limpeza de divisão em um mesmo local
# Caso quisermos apenas dividir? OU se quisermos apenas limpar?
# Como garantimos que os dados foram limpos corretamente?


def limpar_separar_e_dividir_dados(
    dataframe: pd.DataFrame, coluna_alvo: str, test_size: float = 0.2
):
    df_processado = dataframe.copy()
    colunas_para_remover = ["nome", "idade"]
    df_processado = df_processado.drop(columns=colunas_para_remover)

    X = df_processado.drop(columns=[coluna_alvo])
    y = df_processado[coluna_alvo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


# EXEMPLO SEGUINDO O PRINCÍPIO DA RESPONSABILIDADE ÚNICA

# Aqui temos funções separadas para cada responsabilidade.


def limpar_dados(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_limpo = dataframe.copy()
    colunas_para_remover = ["nome", "idade"]
    df_limpo = df_limpo.drop(columns=colunas_para_remover)

    return df_limpo


def separar_features_e_alvo(
    dataframe: pd.DataFrame, coluna_alvo: str
) -> tuple[pd.DataFrame, pd.Series]:
    X = dataframe.drop(columns=[coluna_alvo])
    y = dataframe[coluna_alvo]

    return X, y


def dividir_em_treino_e_teste(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


# Exemplo de uso

if __name__ == "__main__":
    dados_exemplo = pd.DataFrame(
        {
            "nome": ["Alice", "Bob", "Charlie", "David"],
            "idade": [25, 30, 35, 40],
            "renda": [50000, 60000, 70000, 80000],
            "comprou": [0, 1, 0, 1],
        }
    )

# Usando a função que viola o princípio
# Como sabemos se o X_train já está limpo olhando apenas essa parte do código?

X_train_v1, X_test_v1, y_train_v1, y_train_v1 = limpar_separar_e_dividir_dados(
    dados_exemplo, coluna_alvo="comprou"
)
print("Exemplo sem SRP:")
print("X_train: \n", X_train_v1)
print("y_train: \n", y_train_v1)

# Usando as funções que seguem o princípio
dados_limpos = limpar_dados(dados_exemplo)
X, y = separar_features_e_alvo(dados_limpos, coluna_alvo="comprou")
X_train_v2, X_test_v2, y_train_v2, y_test_v2 = dividir_em_treino_e_teste(X, y)

print("Exemplo com SRP:")
print("X_train: \n", X_train_v2)
print("y_train: \n", y_train_v2)
