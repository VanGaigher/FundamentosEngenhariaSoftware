"""
Módulo responsável pelo treinamento do modelo de classificação.

Este módulo carrega os dados de treino já pré-processados, seleciona as
principais variáveis preditoras definidas em ``TOP_FEATURES`` e treina um
modelo de Árvore de Decisão com hiperparâmetros fixos. Após o ajuste, o
modelo treinado é serializado e salvo em disco para uso posterior na
etapa de avaliação.
"""

import logging

import pandas as pd
from joblib import dump
from sklearn.tree import DecisionTreeClassifier


from fundamentos_engenharia_software.config import TOP_FEATURES

logger = logging.getLogger(__name__)


def train_model(
    x_train_imputed_data_path: str, y_train_data_path: str, model_path: str
) -> None:
    """
    Treina um modelo de Árvore de Decisão e salva o artefato resultante.

    O processo envolve:

    * carregar os conjuntos de treino (features e target);
    * selecionar as variáveis definidas em ``TOP_FEATURES``;
    * instanciar o classificador com hiperparâmetros pré-definidos;
    * ajustar o modelo aos dados;
    * serializar o modelo treinado e salvá-lo em ``model_path``.

    :param str x_train_imputed_data_path:
        Caminho para o arquivo CSV contendo as features de treino já imputadas.

    :param str y_train_data_path:
        Caminho para o arquivo CSV contendo a variável target de treino.

    :param str model_path:
        Caminho onde o artefato do modelo treinado será salvo (formato ``.joblib``).

    :raises FileNotFoundError:
        Caso algum dos arquivos de treino não seja encontrado.

    :raises Exception:
        Para quaisquer erros inesperados durante o processo de treinamento.

    :returns:
        ``None``. O modelo é persistido em disco.
    """
    try:
        logger.info("Iniciando o treinamento do modelo.")

        X_train = pd.read_csv(x_train_imputed_data_path)
        y_train = pd.read_csv(y_train_data_path)

        X_train_top = X_train[TOP_FEATURES]

        dt_model_top = DecisionTreeClassifier(
            criterion="entropy",
            random_state=42,
            max_depth=6,
            min_samples_leaf=20,
            min_samples_split=20,
            class_weight="balanced",
        )

        dt_model_top.fit(X_train_top, y_train)
        logger.info("Salvando o modelo treinado em %s", model_path)

        dump(dt_model_top, model_path)
        logger.info("Modelo final salvo em 'modelo_final.joblib'.")
    except FileNotFoundError as e:
        logger.error("Arquivo de dados de treino não encontrado: %s", e)
        raise
    except Exception as e:
        logger.error("Ocorreu um erro durante o treinamento: %s", e)
        raise
