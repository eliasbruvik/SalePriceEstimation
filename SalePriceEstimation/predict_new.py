from data_preprocessor import DataPreprocessor
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import joblib
from pickle import load
from nptyping import NDArray


class Predictor:
    """A class used to preprocess and predict 'new' data"""

    def __init__(self, *, model_directory: str) -> None:
        self._load_preprocessor(model_directory)
        self._load_model(model_directory)

    def predict(self, df: pd.DataFrame) -> NDArray:
        """Pre-processes new data and calculates the prediction"""
        processed_data = self.data_preprocessor.transform_new_data(df)
        prediction = self.model.predict(processed_data)
        return prediction.reshape((processed_data.shape[0],))

    def _load_preprocessor(self, model_directory: str) -> None:
        """Loads the data preprocessor"""
        preprocessor_directory = os.path.join(model_directory, "data_preprocessor")
        data_preprocessor = DataPreprocessor()
        data_preprocessor.load(preprocessor_directory)
        self.data_preprocessor = data_preprocessor

    def _load_model(self, model_directory: str) -> None:
        """Loads a pre-trained model"""
        settings = load(open(os.path.join(model_directory, "settings.pkl"), "rb"))
        if settings["loader"] == "keras":
            self.model = tf.keras.models.load_model(
                os.path.join(model_directory, "model")
            )
        elif settings["loader"] == "joblib":
            self.model = joblib.load(os.path.join(model_directory, "model.joblib"))


if __name__ == "__main__":
    np.random.seed(0)
    X = pd.read_csv("data/TrainAndValid.csv").sample(frac=1)[:20]
    y = X.pop("SalePrice")
    model_directory = "data\\models\\simple_neural"
    predictor = Predictor(model_directory=model_directory)
    prediction = predictor.predict(X)
    for i in range(len(prediction)):
        print(f"prediction: {prediction[i]}, solution: {y.values[i]}")
