'''

This file includes the tests
for ___________, results
of the test logs are found in ./logs/test.log

Author: Tesnim Hadhri
Date: 

'''

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import logging
from shutil import ExecError
from train_model import read_return_train_test_data, train_model

def test_clean_data():
    assert os.path.isfile('./data/census.csv')

def test_read_return_train_test_data():
    train, test = read_return_train_test_data(f"./data/clean_data.csv")
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
    train_model(cat_features, train)
    assert os.path.isfile("./model/model.joblib")
    assert os.path.isfile("./model/encoder.joblib")
    assert os.path.isfile("./model/lb.joblib")

# def test_score_per_slice():
    # pass
