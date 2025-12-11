import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, "raw")
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, "processed")
ARTIFACTS_FOLDER = os.path.join(PROJECT_ROOT, "artifacts")

RAW_DATA_PATH = os.path.join(RAW_DATA_FOLDER, "dados.xlsx")
PROCESSED_DATA_PATH = os.path.join(
    PROCESSED_DATA_FOLDER, "dados_com_features.csv"
)
MODEL_PATH = os.path.join(ARTIFACTS_FOLDER, "modelo_treinado.joblib")

X_TRAIN_PATH = os.path.join(PROCESSED_DATA_FOLDER, "X_train_imputed.csv")
Y_TRAIN_PATH = os.path.join(PROCESSED_DATA_FOLDER, "y_train.csv")
X_TEST_PATH = os.path.join(PROCESSED_DATA_FOLDER, "X_test_imputed.csv")
Y_TEST_PATH = os.path.join(PROCESSED_DATA_FOLDER, "y_test.csv")

COLS_TO_USE = [
    "score_1",
    "score_2",
    "score_3",
    "score_4",
    "score_5",
    "score_6",
    "score_7",
    "score_8",
    "score_9",
    "score_10",
    "entrega_doc_1",
    "entrega_doc_3",
    "valor_compra",
    "entrega_doc_2_nan",
    "paises_agrupados",
    "grupo_categorias",
    "entrega_doc",
    "fraude",
]

TOP_FEATURES = [
    "entrega_doc_2_nan",
    "score_10",
    "grupo_categorias",
    "entrega_doc_1",
    "score_6",
    "score_9",
    "score_1",
    "score_7",
    "valor_compra",
    "score_4",
    "score_2",
    "entrega_doc_3",
    "score_3",
]
