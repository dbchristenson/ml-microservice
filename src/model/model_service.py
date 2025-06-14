"""
This module contains the ModelService class, which is responsible for
loading, training, and predicting with a machine learning model.

It uses the RandomForestRegressor from sklearn and the Loguru library
for logging. The model is loaded from a specified path, and if it does
not exist, a new model is trained and saved. The class also provides
a method for making predictions using the loaded model.
"""

import os
import pickle as pk
from pathlib import Path

from loguru import logger
from sklearn.ensemble import RandomForestRegressor

from src.config.config import settings
from src.model.pipeline.model import build_model


class ModelService:
    """
    Class to handle model service, loading, training, and predicting.

    This class loads a pre-trained model from disk if the configured model path
    exists. If the model does not exist, then the class will train a new model
    under the given name.

    Attributes:
        model: RandomForestRegressor

    Methods:
        load_model: Load the model from disk.
        predict: Make predictions using the loaded model.
    """

    def __init__(self):
        self.model = None

    def load_model(self):
        """
        Load the model from disk.

        If the model does not exist, it will train a new model.
        """
        model_path = Path(
            os.path.join(settings.model_path, settings.model_name)
        )

        if not model_path.exists():
            logger.warning("Model not found, training a new model.")
            build_model(model_name=settings.model_name)

        logger.info("Model exists -> loading model.")

        try:
            with open(model_path, "rb") as model_file:
                self.model: RandomForestRegressor = pk.load(model_file)
        except Exception as exception:
            logger.critical(f"Error loading model: {exception}")
            raise

        logger.info("Model loaded successfully.")

    def predict(self, input_parameters: list) -> float:
        """
        Make predictions using the loaded model.

        Args:
            input_parameters (list): List of input parameters for prediction.

        Returns:
            float: Predicted value.
        """
        logger.debug(f"Input parameters for prediction: {input_parameters}")
        return self.model.predict([input_parameters])
