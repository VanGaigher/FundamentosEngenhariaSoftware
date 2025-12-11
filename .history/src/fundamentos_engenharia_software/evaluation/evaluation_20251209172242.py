import numpy as np

"""
Módulo de avaliação de modelo.

Este módulo contém funções para carregar um modelo treinado e dados,
realizar predições e calcular métricas de desempenho como AUC, KS,
precisão e recall.

As principais funções são:

- evaluate_model: Orquestra o processo completo, chamando as outras funções em sequência.
- load_data_and_artifacts: Carrega o modelo serializado e os DataFrames de treino/teste.
- make_predictions: Utiliza o modelo para gerar predições nos dados.
- calculate_metrics: Calcula e exibe as métricas de avaliação do modelo.
"""

from typing import Any, Dict, Tuple
import joblib
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.fundamentos_engenharia_software.config import (
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    TOP_FEATURES,
    MODEL_PATH,
)


# Definição da classe ModelEvaluator:
class ModelEvaluator:
    """
    :param model_path: Caminho para o arquivo do modelo (.joblib).
    :type model_path: str
    :param x_train_path: Caminho para o CSV de features de treino.
    :type x_train_path: str
    :param x_test_path: Caminho para o CSV de features de teste.
    :type x_test_path: str
    :param y_train_path: Caminho para o CSV de alvos de treino.
    :type y_train_path: str
    :param y_test_path: Caminho para o CSV de alvos de teste.
    :type y_test_path: str
    :param top_features: Lista com os nomes das features mais importantes.
    :type top_features: list

    :ivar model: O modelo de machine learning carregado.
    :vartype model: Any
    :ivar X_test: DataFrame com os dados de teste.
    :vartype X_test: pandas.DataFrame
    :ivar y_test: Series com os alvos de teste.
    :vartype y_test: pandas.Series
    :ivar metrics: Dicionário com os resultados da avaliação (AUC, KS, etc.).
    :vartype metrics: dict[str, float]
    """

    # Inicialização da classe com os caminhos dos artefatos e dados
    def __init__(
        self,
        model_path: str,
        x_train_path: str,
        x_test_path: str,
        y_train_path: str,
        y_test_path: str,
        top_features: list,
    ):
        self.model_path = model_path
        self.x_train_path = x_train_path
        self.x_test_path = x_test_path
        self.y_train_path = y_train_path
        self.y_test_path = y_test_path
        self.top_features = top_features
        self.model: Any = None
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        self.y_train_proba: np.ndarray = None
        self.y_pred_proba: np.ndarray = None
        self.y_pred: np.ndarray = None
        self.metrics: dict[str, float] = {}

    def load_data_and_artifacts(self) -> None:
        """
        Carrega o modelo treinado e os conjuntos de dados de treino e teste.

        Lê o modelo serializado (joblib) e os arquivos CSV contendo os dados
        processados para treino e teste.
        """
        try:
            print(f"Carregando modelo de {self.model_path}")
            self.model = joblib.load(
                self.model_path
            )  # atribui o modelo carregado à variável de instância
            self.X_train = pd.read_csv(
                self.x_train_path
            )  # atribui os dados de treino à variável de instância
            self.y_train = pd.read_csv(
                self.y_train_path
            ).squeeze()  # atribui os alvos de treino à variável de instância
            self.X_test = pd.read_csv(
                self.x_test_path
            )  # atribui os dados de teste à variável de instância
            self.y_test = pd.read_csv(
                self.y_test_path
            ).squeeze()  # atribui os alvos de teste à variável de instância

        except FileNotFoundError as e:
            print(f"Arquivo de modelo ou de dados não encontrado: {e}")
            raise
        except Exception as e:
            print(f"Falha ao carregar artefatos: {e}")
            raise

    def make_predictions(self) -> None:
        """
        Gera predições usando o modelo treinado.

        Calcula as probabilidades preditas para os conjuntos de treino e teste,
        e as classes preditas para o conjunto de teste, utilizando apenas as
        features mais importantes (TOP_FEATURES).

        :param model: O modelo de machine learning treinado e carregado.
        :type model: object
        :param X_train: DataFrame com as features de treino.
        :type X_train: pd.DataFrame
        :param X_test: DataFrame com as features de teste.
        :type X_test: pd.DataFrame

        """

        self.y_train_proba = self.model.predict_proba(
            self.x_train_path[self.top_features]
        )[:, 1]
        self.y_pred_proba = self.model.predict_proba(
            self.x_test_path[self.top_features]
        )[:, 1]
        self.y_pred = self.model.predict(self.x_test_path[self.top_features])

    def calculate_metrics(self) -> None:
        """
        Calcula e exibe as métricas de avaliação do modelo.

        Temos as métricas:
        - ROC AUC (para treino e teste)
        - Estatística KS
        - Precisão e recall (para teste)

        """
        ks_stat, ks_p = ks_2samp(
            self.y_pred_proba[self.y_test == 1],
            self.y_pred_proba[self.y_test == 0],
        )

        # Métricas
        self.metrics = {
            "AUC_Train": roc_auc_score(self.y_train, self.y_train_proba),
            "AUC_Test": roc_auc_score(self.y_test, self.y_pred_proba),
            "KS_Statistic": ks_stat,
            "KS_P_value": ks_p,
            "Precision": precision_score(self.y_test, self.y_pred),
            "Recall": recall_score(self.y_test, self.y_pred),
        }

    def evaluate_model(self) -> Dict[str, float]:
        """
        Orquestra o processo completo de avaliação do modelo.

        Esta função serve como nossa "cola" das etapas para a avaliação,
        chamando as funções para carregar dados, fazer predições e calcular
        as métricas.
        """
        self._load_data_and_artifacts()
        self._make_predictions()
        self._calculate_metrics()
        return self.metrics


def evaluate_model() -> None:
    """
    Função principal para avaliar o modelo.

    Esta função instancia a classe ModelEvaluator com os caminhos dos
    artefatos e dados, e executa o processo de avaliação, exibindo as
    métricas calculadas.
    """
    evaluator = ModelEvaluator(
        model_path=MODEL_PATH,
        x_train_path=X_TRAIN_PATH,
        x_test_path=X_TEST_PATH,
        y_train_path=Y_TRAIN_PATH,
        y_test_path=Y_TEST_PATH,
        top_features=TOP_FEATURES,
    )

    metrics = evaluator.evaluate_model()
    print("Métricas de Avaliação do Modelo:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
