# mypy: ignore-errors
import numpy as np
import flask
from predict_new import Predictor
import pandas as pd
import json
import os
from typing import Dict


"""Model params"""
default_model = "random_forest"
models_directory = "data\\models"


def load_models(models_directory: str) -> Dict[str, Predictor]:
    predictors = {}
    dirs = os.listdir(models_directory)
    for model in dirs:
        predictors[model] = Predictor(
            model_directory=os.path.join(models_directory, model)
        )
    return predictors


np.random.seed(0)
app = flask.Flask(__name__)
predictors = load_models(models_directory)


@app.route("/predict", methods=["POST"])
def predict() -> flask.Response:
    response = {}
    status = 400
    if flask.request.method == "POST":
        model = flask.request.args.get("model", default_model)
        if "data" in flask.request.files.keys() and model in predictors.keys():
            df = pd.read_csv(flask.request.files["data"])
            prediction = predictors[model].predict(df)
            response["predictions"] = prediction.tolist()
            status = 200
    return flask.Response(json.dumps(response), status)


if __name__ == "__main__":
    app.run()
