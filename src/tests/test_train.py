import pytest
from pathlib import Path
import os
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from mlem.api import load
import sys

from src.scripts.train import train_diabetes_model


@pytest.fixture
def config():
    with open("configs.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def test_train_diabetes_model_no_nans(config, params, tmp_path):
    config["train"]["model_file"] = tmp_path / "model.pkl"
    train_diabetes_model(config, params)

    # Check that model file exists
    assert os.path.exists(config["train"]["model_file"])
    model = load(config["train"]["model_file"])

    # Check that predictions are not NaNs
    X_train = pd.read_csv(config["data_clean"]["X_train"])
    assert not X_train.isnull().values.any()
    y_train = pd.read_csv(config["data_clean"]["y_train"])
    assert not y_train.isnull().values.any()

    assert not any(
        map(lambda x: x != x, model.predict(X_train))
    )  # Check that predictions are not NaNs
