import pandas as pd
import numpy as np

# DataFrame de TREINO (histórico, com muitas transações)

data_treino = {
    "id_transacao": range(10),
    "categoria_produto": [
        "eletrônicos",
        "moda",
        "casa e jardim",
        "eletrônicos",
        "livros",
        "moda",
        "eletrônicos",
        "brinquedos",
        "moda",
        "esportes",
    ],
    "valor": [1200, 150, 300, 4500, 45, 200, 800, 90, 250, 500],
}
df_treino = pd.DataFrame(data_treino)

# DataFrame de INFERÊNCIA (novos dados que acabaram de chegar)

data_inferencia = {
    "id_transacao": range(10, 14),
    "categoria_produto": ["eletrônicos", "livros", "moda", "saúde"],
    "valor": [950, 60, 180, 120],
}
df_inferencia = pd.DataFrame(data_inferencia)

# EXEMPLO DE CÓDIGO REPETIDO COM VIOLAÇÃO DO DRY
contagem_categorias = df_treino["categoria_produto"].value_counts()
categoria_raras = contagem_categorias[contagem_categorias > 2].index.tolist()
print(f"categorias raras identificadas no treino:{categoria_raras}")

df_treino_processado_v1 = df_treino.copy()
df_treino_processado_v1["grupo_categorias"] = df_treino_processado_v1[
    "categoria_produto"
]
df_treino_processado_v1.loc[
    df_treino_processado_v1["grupo_categorias"].isin(categoria_raras),
    "grupo_categorias",
] = "outras_categorias"

df_inferencia_processado_v1 = df_inferencia.copy()
df_inferencia_processado_v1["grupo_categorias"] = df_inferencia_processado_v1[
    "categoria_produto"
]
df_inferencia_processado_v1.loc[
    df_inferencia_processado_v1["grupo_categorias"].isin(categoria_raras),
    "grupo_categorias",
] = "outras_categorias"


# EXEMPLO DE CÓDIGO SEGUINDO O PRINCÍPIO DRY


def agrupar_categorias_raras(
    df: pd.DataFrame, lista_categorias_raras: list
) -> pd.DataFrame:
    df_processado = df.copy()
    df_processado["grupo_categorias"] = df_processado["categoria_produto"]
    df_processado.loc[
        df_processado["grupo_categorias"].isin(lista_categorias_raras),
        "grupo_categorias",
    ] = "outras_categorias"

    return df_processado


df_treino_processado_v2 = agrupar_categorias_raras(df_treino, categoria_raras)

df_inferencia_processado_v2 = agrupar_categorias_raras(df_inferencia, categoria_raras)
