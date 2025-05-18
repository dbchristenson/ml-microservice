from loguru import logger

from src.model.model_service import ModelService


@logger.catch
def main():
    ml_svc = ModelService()
    ml_svc.load_model()
    prediction = ml_svc.predict([85, 2015, 2, 2, 1, 20, 1, 1, 0, 0, 1])
    logger.info(f"Prediction: {prediction}")

    return prediction


if __name__ == "__main__":
    main()
