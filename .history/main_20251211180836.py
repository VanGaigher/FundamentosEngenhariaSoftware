from src.fundamentos_engenharia_software.preprocessing.feature_engineering import (
    run_feature_engineer,
)
from src.fundamentos_engenharia_software.preprocessing.impute_and_scale import (
    impute_and_scale,
)
from src.fundamentos_engenharia_software.training.modeling import (
    train_model,
)
from src.fundamentos_engenharia_software.evaluation.evaluation import (
    evaluate_model,
)


def main():
    try:
        run_feature_engineer()
        impute_and_scale()

        train_model()
        evaluate_model()
    except Exception as e:
        print(f"Erro durante o processamento: {e}")


if __name__ == "__main__":
    main()
