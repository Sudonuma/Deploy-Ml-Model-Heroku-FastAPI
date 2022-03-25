'''

This file includes the tests
for ___________, results
of the test logs are found in ./logs/test.log

Author: Tesnim Hadhri
Date: 

'''

import os
import logging
from shutil import ExecError
import starter.train_model

def test_clean_data():
    assert os.path.isfile('/root/Deploy-Ml-Model-Heroku-FastAPI/data/census.csv.dvc')

def test_read_return_train_test_data():
    train, test = starter.train_model.read_return_train_test_data(f"/root/Deploy-Ml-Model-Heroku-FastAPI/data/clean_data.csv")
    assert train.shape[0] > 0
    assert test.shape[0] > 0

    return train, test

def test_train_model():
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    train, _ = test_read_return_train_test_data()
    starter.train_model.train_model(cat_features, train)
    assert os.path.isfile("/root/Deploy-Ml-Model-Heroku-FastAPI/model/model.joblib")
    assert os.path.isfile("/root/Deploy-Ml-Model-Heroku-FastAPI/model/encoder.joblib")
    assert os.path.isfile("/root/Deploy-Ml-Model-Heroku-FastAPI/model/lb.joblib")

# def test_score_per_slice():
    # pass