from fundamentos_engenharia_software.preprocessing.feature_engineering import (
    FeatureEngineer,
)
from fundamentos_engenharia_software.preprocessing.impute_and_scale import (
    DataScalerAndImputer,
)

from fundamentos_engenharia_software.training.modeling import train_model
from fundamentos_engenharia_software.evaluation.evaluation import (
    ModelEvaluator,
)

from fundamentos_engenharia_software.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    X_TRAIN_PATH,
    Y_TRAIN_PATH,
    PROCESSED_DATA_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
    COLS_TO_USE,
    MODEL_PATH,
    TOP_FEATURES,
    ARTIFACTS_FOLDER,
)


def run_training():
    try:
        feature_engineer = FeatureEngineer(
            input_path=RAW_DATA_PATH,
            output_path=PROCESSED_DATA_PATH,
            artifacts_path=ARTIFACTS_FOLDER,
        )

        feature_engineer.run_feature_engineer()

        scaler_and_imputer = DataScalerAndImputer(
            input_path=PROCESSED_DATA_PATH,
            output_x_train_path=X_TRAIN_PATH,
            output_x_test_path=X_TEST_PATH,
            output_y_train_path=Y_TRAIN_PATH,
            output_y_test_path=Y_TEST_PATH,
            cols_to_use=COLS_TO_USE,
            artifacts_path=ARTIFACTS_FOLDER,
        )

        scaler_and_imputer.run()

        train_model(X_TRAIN_PATH, Y_TRAIN_PATH, MODEL_PATH)

        evaluator = ModelEvaluator(
            model_path=MODEL_PATH,
            x_train_path=X_TRAIN_PATH,
            x_test_path=X_TEST_PATH,
            y_train_path=Y_TRAIN_PATH,
            y_test_path=Y_TEST_PATH,
            top_features=TOP_FEATURES,
        )

        final_metrics = evaluator.evaluate()
        print(f"Dicionário de métricas: {final_metrics}")

    except Exception as e:
        print(f"Erro durante a execução do pipeline: {e}")


if __name__ == "__main__":
    run_training()
