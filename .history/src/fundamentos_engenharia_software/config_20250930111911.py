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
