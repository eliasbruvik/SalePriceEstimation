from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any, List, Dict, Tuple, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pickle import dump, load
import os


class DataPreprocessor:
    def __init__(
        self, df: pd.DataFrame = None, *, target: str = "", test_percentage: float = 0.0
    ) -> None:
        self.df = df
        if target:
            self.df = self.df.sample(frac=1)
            self.y = self.df.pop(target)
            self.train_size = int((1.0 - test_percentage) * len(df))

        # Data and models
        self.steps: List[str] = []
        self.drop_features_save: List[str] = []
        self.nan_values_save: Dict[str, List[Any]] = {}
        self.date_features_save: List[str] = []
        self.scaler_save = None
        self.scale_features_save: List[str] = []
        self.onehot_features_save: List[str] = []
        self.onehot_encoder_save = None
        self.categorical_imputer_save = None
        self.categorical_features_save: List[str] = []
        self.continuous_imputer_save = None
        self.continuous_features_save: List[str] = []

    def save(self, directory: str) -> None:
        """Saves all necessary data to transform new data later"""
        os.makedirs(directory, exist_ok=True)
        dump(self.steps, open(os.path.join(directory, "steps.pkl"), "w+b"))
        dump(
            self.drop_features_save,
            open(os.path.join(directory, "drop_features.pkl"), "w+b"),
        )
        dump(
            self.nan_values_save, open(os.path.join(directory, "nan_values.pkl"), "w+b")
        )
        dump(
            self.date_features_save,
            open(os.path.join(directory, "date_features.pkl"), "w+b"),
        )
        dump(self.scaler_save, open(os.path.join(directory, "scaler.pkl"), "w+b"))
        dump(
            self.scale_features_save,
            open(os.path.join(directory, "scale_features.pkl"), "w+b"),
        )
        dump(
            self.onehot_features_save,
            open(os.path.join(directory, "onehot_features.pkl"), "w+b"),
        )
        dump(
            self.onehot_encoder_save,
            open(os.path.join(directory, "onehot_encoder.pkl"), "w+b"),
        )
        dump(
            self.categorical_imputer_save,
            open(os.path.join(directory, "categorical_imputer.pkl"), "w+b"),
        )
        dump(
            self.categorical_features_save,
            open(os.path.join(directory, "categorical_features.pkl"), "w+b"),
        )
        dump(
            self.continuous_imputer_save,
            open(os.path.join(directory, "continuous_imputer.pkl"), "w+b"),
        )
        dump(
            self.continuous_features_save,
            open(os.path.join(directory, "continuous_features.pkl"), "w+b"),
        )

    def load(self, directory: str) -> None:
        """Loads previously fit data for use when transforming new data"""
        self.steps = load(open(os.path.join(directory, "steps.pkl"), "rb"))
        self.drop_features_save = load(
            open(os.path.join(directory, "drop_features.pkl"), "rb")
        )
        self.nan_values_save = load(
            open(os.path.join(directory, "nan_values.pkl"), "rb")
        )
        self.date_features_save = load(
            open(os.path.join(directory, "date_features.pkl"), "rb")
        )
        self.scaler_save = load(open(os.path.join(directory, "scaler.pkl"), "rb"))
        self.scale_features_save = load(
            open(os.path.join(directory, "scale_features.pkl"), "rb")
        )
        self.onehot_features_save = load(
            open(os.path.join(directory, "onehot_features.pkl"), "rb")
        )
        self.onehot_encoder_save = load(
            open(os.path.join(directory, "onehot_encoder.pkl"), "rb")
        )
        self.categorical_imputer_save = load(
            open(os.path.join(directory, "categorical_imputer.pkl"), "rb")
        )
        self.categorical_features_save = load(
            open(os.path.join(directory, "categorical_features.pkl"), "rb")
        )
        self.continuous_imputer_save = load(
            open(os.path.join(directory, "continuous_imputer.pkl"), "rb")
        )
        self.continuous_features_save = load(
            open(os.path.join(directory, "continuous_features.pkl"), "rb")
        )

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms new data using the saved steps"""
        old_df = self.df
        self.df = df
        steps = self.steps[:]
        for step in steps:
            if step == "drop_features":
                self.drop_features(self.drop_features_save)
            elif step == "convert_to_nan":
                self.convert_to_nan(self.nan_values_save)
            elif step == "datestr_to_ordinal":
                self.datestr_to_ordinal(self.date_features_save)
            elif step == "scale_features":
                self.scale_features(self.scale_features_save)
            elif step == "onehot_encode_features":
                self.onehot_encode_features(self.onehot_features_save)
            elif step == "impute_missing_values":
                self.impute_missing_values(
                    categorical_features=self.categorical_features_save,
                    continuous_features=self.continuous_features_save,
                )
        transformed_data = self.df
        self.df = old_df
        self.steps = steps
        return transformed_data

    def drop_features(self, drop_features: List[str]) -> None:
        """Drops unwanted features from the data frame"""
        self.df = self.df.drop(columns=drop_features)

        self.steps.append("drop_features")
        if not self.drop_features_save:
            self.drop_features_save = drop_features

    def convert_to_nan(self, nan_values: Dict[str, List[Any]]) -> None:
        """Converts values for each feature that should be interpreted as NaN (not specified)

        nan_values: Dict of format {feature: [list of nan-values for this feature]}
            Using the feature 'All' converts all instances in the data frame.
        """
        for feature, missing_values in nan_values.items():
            for missing_value in missing_values:
                if feature == "All":
                    self.df = self.df.replace(missing_value, np.nan)
                else:
                    self.df[feature] = self.df[feature].replace(missing_value, np.nan)

        self.steps.append("convert_to_nan")
        if not self.nan_values_save:
            self.nan_values_save = nan_values

    def datestr_to_ordinal(self, date_features: List[str]) -> None:
        """Converts date features to ordinal format"""
        for date_feature in date_features:
            self.df[date_feature] = (
                self.df[date_feature]
                .astype("datetime64[ns]")
                .apply(lambda date: date.toordinal())
            )

        self.steps.append("datestr_to_ordinal")
        if not self.date_features_save:
            self.date_features_save = date_features

    def scale_features(self, scale_features: List[str]) -> None:
        """Scales features"""
        if self.scaler_save is not None:
            scaler = self.scaler_save
        else:
            scaler = StandardScaler()
            scaler.fit(self.df[scale_features][: self.train_size])
        self.df[scale_features] = scaler.transform(self.df[scale_features])

        self.steps.append("scale_features")
        if self.scaler_save is None:
            self.scaler_save = scaler
            self.scale_features_save = scale_features

    def onehot_encode_features(self, onehot_features: List[str]) -> None:
        """Onehot-encodes categorical features"""
        df_onehot = self.df[onehot_features].copy()
        self.df = self.df.drop(onehot_features, axis=1)
        # df_rest = self.df[list(set(self.df.columns) - set(onehot_features))]
        if self.onehot_encoder_save is not None:
            encoder = self.onehot_encoder_save
        else:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            encoder.fit(df_onehot[: self.train_size])
        df_categorical = pd.DataFrame(
            encoder.transform(df_onehot), columns=encoder.get_feature_names()
        )
        # df_rest.index = df_categorical.index
        df_categorical.index = self.df.index
        self.df = df_categorical.join(self.df)

        self.steps.append("onehot_encode_features")
        if self.onehot_encoder_save is None:
            self.onehot_features_save = onehot_features
            self.onehot_encoder_save = encoder

    def impute_missing_values(
        self,
        *,
        categorical_features: List[str],
        continuous_features: List[str],
        categorical_strategy: Optional[str] = None,
        continuous_strategy: Optional[str] = None,
    ) -> None:
        """Uses simple imputing methods where there is missing data"""
        self.df[categorical_features] = self.df[categorical_features].astype(
            "category"
        )  # TODO: Sjekk om denne er nødvendig!!

        if self.categorical_imputer_save is not None:
            categorical_imputer = self.categorical_imputer_save
        else:
            categorical_imputer = SimpleImputer(
                strategy=categorical_strategy, fill_value="unspecified"
            )
            categorical_imputer.fit(self.df[categorical_features][: self.train_size])
        self.df[categorical_features] = categorical_imputer.transform(
            self.df[categorical_features]
        )
        self.df[categorical_features] = self.df[categorical_features].astype("str")

        if self.continuous_imputer_save is not None:
            continuous_imputer = self.continuous_imputer_save
        else:
            continuous_imputer = SimpleImputer(strategy=continuous_strategy)
            continuous_imputer.fit(self.df[continuous_features][: self.train_size])
        self.df[continuous_features] = continuous_imputer.transform(
            self.df[continuous_features]
        )

        self.steps.append("impute_missing_values")
        if self.continuous_imputer_save is None:
            self.continuous_imputer_save = continuous_imputer
            self.continuous_features_save = continuous_features
            self.categorical_imputer_save = categorical_imputer
            self.categorical_features_save = categorical_features

    def train_test_split(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns the dataset split into X_train, X_test, y_train, y_test"""
        X_train: pd.DataFrame = self.df[: self.train_size]
        X_test: pd.DataFrame = self.df[self.train_size :]
        y_train: pd.DataFrame = self.y[: self.train_size]
        y_test: pd.DataFrame = self.y[self.train_size :]
        return X_train, X_test, y_train, y_test
