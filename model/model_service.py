import os
import pickle as pk
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from model import build_model


class ModelService:
    def __init__(self):
        self.model = None

    def load_model(self, model_name="rf_v1"):
        model_base_path = "model/models"
        model_path = Path(os.path.join(model_base_path, model_name))

        if not model_path.exists():
            build_model(model_name=model_name)

        self.model: RandomForestRegressor = pk.load(open(model_path, "rb"))

    def predict(self, input_parameters):
        return self.model.predict([input_parameters])


if __name__ == "__main__":
    ml_svc = ModelService()
    ml_svc.load_model("rf_v3")
    pred = ml_svc.predict([85, 2015, 2, 2, 1, 20, 1, 1, 0, 0, 1])
    print(f"Predicted Rental Price: {pred}")
