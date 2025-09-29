import pandas as pd

# EXEMPLO DE FUNÇÃO IMPURA


def preparar_dados_impura(dataframe_parametro: pd.DataFrame):
    dataframe_parametro.dropna(inplace=True)
    dataframe_parametro["price_per_item"] = (
        dataframe_parametro["total_price"] / dataframe_parametro["items"]
    )

    # Exta função não precisa de um "return" para causar o dano


# EXEMPLO FUNÇÃO PURA


def preparar_dados_pura(dataframe_parametro: pd.DataFrame):
    df_copia = dataframe_parametro.copy()
    df_copia.dropna(inplace=True)
    df_copia["price_per_item"] = df_copia["total_price"] / df_copia["items"]

    return df_copia


if __name__ == "__main__":
    dados_exemplo_1 = pd.DataFrame(
        {
            "total_price": [100, 200, None, 150],
            "items": [2, 4, 3, None],
        }
    )

    dados_exemplo_2 = dados_exemplo_1.copy()

    print("\nDados Originais:")
    print(dados_exemplo_1)
    preparar_dados_impura(dados_exemplo_1)
    print("\n Dados Após a Função Impura:")
    print(dados_exemplo_1)

    print("Dados Originais:")
    print(dados_exemplo_2)
    preparar_dados_pura(dados_exemplo_2)
    print("\n Dados Após a Função Pura:")
    print(dados_exemplo_2)
