from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from typing import List, Dict, Any
from data_preprocessor import DataPreprocessor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from pickle import dump
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib


""" Dataset params """
csv_file = "data/TrainAndValid.csv"
models_base_dir = "data\\models"
nan_values: Dict[
    str, List[Any]
] = {  # Values for each feature that should be interpreted as NaN (not specified)
    "All": ["Unspecified", "unspecified", "None or Unspecified"],
    "YearMade": [1000],
    "MachineHoursCurrentMeter": [0],
}
continuous_features = [
    "YearMade",
    "MachineHoursCurrentMeter",
    "saledate",
]
categorical_features = [
    "datasource",
    "auctioneerID",
    "UsageBand",
    "fiBaseModel",
    "fiSecondaryDesc",
    "fiModelSeries",
    "fiModelDescriptor",
    "ProductSize",
    "fiProductClassDesc",
    "state",
    "ProductGroup",
    "ProductGroupDesc",
    "Drive_System",
    "Enclosure",
    "Forks",
    "Pad_Type",
    "Ride_Control",
    "Stick",
    "Transmission",
    "Turbocharged",
    "Blade_Extension",
    "Blade_Width",
    "Enclosure_Type",
    "Engine_Horsepower",
    "Hydraulics",
    "Pushblock",
    "Ripper",
    "Scarifier",
    "Tip_Control",
    "Tire_Size",
    "Coupler",
    "Coupler_System",
    "Grouser_Tracks",
    "Hydraulics_Flow",
    "Track_Type",
    "Undercarriage_Pad_Width",
    "Stick_Length",
    "Thumb",
    "Pattern_Changer",
    "Grouser_Type",
    "Backhoe_Mounting",
    "Blade_Type",
    "Travel_Controls",
    "Differential_Type",
    "Steering_Controls",
]
drop_features = ["SalesID", "MachineID", "ModelID", "fiModelDesc"]
date_features = ["saledate"]

df = pd.read_csv(csv_file)


"""Preprocess data"""
np.random.seed(0)
data_preprocessor = DataPreprocessor(df, target="SalePrice", test_percentage=0.2)
data_preprocessor.drop_features(drop_features)
data_preprocessor.convert_to_nan(nan_values)
data_preprocessor.datestr_to_ordinal(date_features)
data_preprocessor.impute_missing_values(
    categorical_features=categorical_features,
    categorical_strategy="constant",
    continuous_features=continuous_features,
    continuous_strategy="median",
)
data_preprocessor.onehot_encode_features(categorical_features)
data_preprocessor.scale_features(continuous_features)

X_train, X_test, y_train, y_test = data_preprocessor.train_test_split()


"""Evaluation metrics"""


def evaluate(model: Any) -> None:
    """Evaluates a model using root mean absolute error"""
    print(f"Train mae: {mean_absolute_error(model.predict(X_train), y_train)}")
    print(f"Test mae: {mean_absolute_error(model.predict(X_test), y_test)}")


def predict_examples(model: Any, samples: int) -> None:
    """Prints prediction and solution examples for a model"""
    to_predict = X_test[:samples]
    solution = y_test[:samples].values
    predictions = model.predict(to_predict).reshape((to_predict.shape[0],))
    for i in range(len(predictions)):
        print(f"Prediction: {predictions[i]}, solution: {solution[i]}")


"""Simple neural model"""
print("\n\n~~~~~~~~~~~~~~Simple neural model~~~~~~~~~~~~~~\n")
model_name = "simple_neural"
model_directory = os.path.join(models_base_dir, model_name)
os.makedirs(model_directory, exist_ok=True)
preprocessor_directory = os.path.join(model_directory, "data_preprocessor")
data_preprocessor.save(preprocessor_directory)

model = Sequential()
model.add(Dense(15, input_shape=X_train.shape, activation="relu"))
model.add(Dense(7, activation="relu"))
model.add(Dense(3, activation="relu"))
model.add(Dense(1, activation="linear"))

optimizer = Adam(lr=0.01)
model.compile(loss="mse", optimizer="adam", metrics=["mse"])

history = model.fit(X_train, y_train, epochs=100, validation_split=0.1)

os.makedirs(model_directory, exist_ok=True)
model.save(os.path.join(model_directory, "model"), overwrite=True)
dump({"loader": "keras"}, open(os.path.join(model_directory, "settings.pkl"), "w+b"))

history_df = pd.DataFrame(history.history)
plt.plot(history_df["loss"], label="loss")
plt.plot(history_df["val_loss"], label="val_loss")
plt.legend()
plt.show()

evaluate(model)
predict_examples(model, 10)


"""Random forest model"""
print("\n\n~~~~~~~~~~~~~~Random forest model~~~~~~~~~~~~~~\n")
model_name = "random_forest"
model_directory = os.path.join(models_base_dir, model_name)
os.makedirs(model_directory, exist_ok=True)
preprocessor_directory = os.path.join(model_directory, "data_preprocessor")
data_preprocessor.save(preprocessor_directory)

split = int(0.9 * len(X_train))
X_valid = X_train[split:].copy()
X_train = X_train[:split].copy()
y_valid = y_train[split:].copy()
y_train = y_train[:split].copy()


model = RandomForestRegressor(
    n_estimators=100,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features=0.5,
    n_jobs=-1,
    verbose=True,
)
model.fit(X_train, y_train)

joblib.dump(model, open(os.path.join(model_directory, "model.joblib"), "w+b"))
dump({"loader": "joblib"}, open(os.path.join(model_directory, "settings.pkl"), "w+b"))

evaluate(model)
predict_examples(model, 10)
