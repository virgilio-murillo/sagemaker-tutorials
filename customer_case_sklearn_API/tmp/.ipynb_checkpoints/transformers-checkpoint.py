import os
import sys
import subprocess

# base and data packages
import json
import glob
import numpy as np
import pandas as pd
import pickle as pk
from typing import Tuple, Union

# sklearn and modeling
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# xgboost
from xgboost import XGBClassifier


class NullFillTransformer(TransformerMixin):
    """
    Auto detects those float-able. Auto null fills based on the minimum value and an offset value:
    -10^offset of minimum value.
    """
    def __init__(self, offset: int=1):
        self.offset = offset

    def fit(self, X: pd.DataFrame, y=None):
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X[X.dtypes.loc[~X.dtypes.astype(str).isin(["object", "bool"])].index]
        floored = np.floor(X.min()).astype("Int64")
        sign = floored.apply(lambda x: -1 if x <= 1 else 1)
        place = floored.apply(np.format_float_scientific).astype(str).apply(lambda x: int(x[-2:]))
        null_fills = sign * (10**(place * sign).apply(lambda x: np.abs(x-self.offset) if x!=0 else x))
        self.null_fills = null_fills.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.replace({"<NA>": np.nan})
        X = X.replace({"nan": np.nan})
        for k, v in self.null_fills.items():
            X[k] = X[k].fillna(v)
        return X


class RawDataProcessor(TransformerMixin):
    """
    Description: Processes the raw data with various recoding and null filling steps.
    Instantiate with features and intervals for dates from the config and run against
    the raw data. To add steps, create a method and add to the sklearn processing
    pipeline under the run method.
    """

    def __init__(self, features: dict):
        self.features = features

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        pipe = Pipeline(
            [
                ("parsing json column", FunctionTransformer(self.parse_json_column)),
                ("fix datatypes", FunctionTransformer(self.fix_dtypes)),
            ]
        )
        X_processed = pipe.transform(X)
        return X_processed

    def parse_yams_segment(self, x):
        if x is None:
            return [np.nan, np.nan]
        else:
            x = (
                x.replace('{"yams_score":', "")
                .replace('"north_star_metric":"', "")
                .replace('"}', "")
            )
            return [x.split(",")[0], x.split(",")[1]]

    def parse_json_column(self, X: pd.DataFrame) -> pd.DataFrame:
        X_split = pd.DataFrame(
            X.in_data.apply(lambda x: self.parse_yams_segment(x)).to_list(),
            columns=["yams_score", "north_star_metric"],
        )
        X = X.drop(columns=["in_data"])
        X = pd.concat([X, X_split], axis=1)
        return X

    def fix_dtypes(self, X: pd.DataFrame) -> pd.DataFrame:
        for i in self.features["continuous"]:
            X[i] = X[i].astype("float64")
        return X


class DataSlicer(TransformerMixin):
    """
    Description: Slices data based on rows (training and testing sets) and also based on
    columns (meta data, features and outcomes). This class is instantiated with the date
    intervals for training and testing and also the features from the config.
    """

    def __init__(self, features: dict, intervals: dict):
        self.features = features
        self.intervals = intervals

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        pipe = Pipeline(
            [
                ("slice columns", FunctionTransformer(self.slice_columns)),
            ]
        )
        X = pipe.transform(X)
        return X

    def slice_dates(self, X) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # slicing training
        training_start_date = pd.to_datetime(X.timestamp) >= pd.to_datetime(
            self.intervals["training_start_date"], format="%Y%m%d", utc=True
        )
        training_end_date = pd.to_datetime(X.timestamp) < pd.to_datetime(
            self.intervals["training_end_date"], format="%Y%m%d", utc=True
        )
        train = X.loc[(training_start_date & training_end_date), :]
        train = train.reset_index(drop=True)
        # slicing testing
        testing_start_date = pd.to_datetime(X.timestamp) >= pd.to_datetime(
            self.intervals["testing_start_date"], format="%Y%m%d", utc=True
        )
        testing_end_date = pd.to_datetime(X.timestamp) < pd.to_datetime(
            self.intervals["testing_end_date"], format="%Y%m%d", utc=True
        )
        test = X.loc[(testing_start_date & testing_end_date), :]
        test = test.reset_index(drop=True)
        return train, test

    def slice_columns(self, X) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # get feature columns in order
        l = self.features["continuous"]
        # get meta columns in order
        meta_cols = [i for i in X.columns if i not in l]
        # further splitting
        d_features = X[l]
        self.d_meta = X[meta_cols]
        return d_features


class FitModel(BaseEstimator):
    """
    Description: Fits a cv hyperparameter tuned model using BaseEstimator. This is
    balled into a pipeline for sm deployment
    """

    def __init__(self, hyper_parameters: dict, folds: int):
        self.hyper_parameters = hyper_parameters
        self.folds = folds
        self.njobs = 8
        self.seed = 1005

    def fit(self, X: pd.DataFrame, y: Union[list, pd.Series, np.array]) -> pd.DataFrame:
        self.model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            scale_pos_weight=20,
            seed=self.seed,
        )

        # init grid search
        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.hyper_parameters,
            scoring="roc_auc",
            n_jobs=self.njobs,
            cv=KFold(n_splits=self.folds),
            verbose=2,
        )

        # fit grid search
        self.grid_search.fit(X, y)

        # init and fit calibration
        self.calibrated_model = CalibratedClassifierCV(
            self.grid_search.best_estimator_,
            method="isotonic",
            cv=self.folds,
            n_jobs=self.njobs,
        )
        self.calibrated_model.fit(X, y.astype(int))

        # adding feature names
        self.grid_search.best_estimator_.get_booster().feature_names = (
            X.columns.to_list()
        )

        return self

    def predict(self, X):
        return {
            "uncalibrated": self.grid_search.best_estimator_.predict_proba(X),
            "calibrated": self.calibrated_model.predict_proba(X),
        }
