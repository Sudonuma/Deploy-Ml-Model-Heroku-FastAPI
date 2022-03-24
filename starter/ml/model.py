from sklearn.metrics import fbeta_score, precision_score, recall_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from .data import process_data
import logging


logging.basicConfig(filename='./logs/category_slice_scores.log', level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.

    """

    # use random forest classifier
    model = RandomForestClassifier(random_state = 42)
    model.fit(X_train, y_train)
    
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : .joblib
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_score_per_slice(model, test, encoder,
                            lb, categorical_features):
    """
    Compute score per slice of category
    inputs
    ------
    model: trained random forest classifier.
    test: test data
    encoder: encoder used for the training
    lb: label binarizer

    Returns
    -------

    """
    for category in categorical_features:
        for key_category in test[category].unique():
            category_df = test[test[category] == key_category]
            x_test, y_test, _, _ = process_data(
                category_df,
                categorical_features=categorical_features, training=False,
                label="salary", encoder=encoder, lb=lb)

            predictions = model.predict(x_test)

            precision, recall, fbeta = compute_model_metrics(y_test, predictions)
            logger.info("category slice: %s : precision: %s, recall %s, fbeta: %s" %(key_category, precision, recall, fbeta))
            print("category slice: %s : precision: %s, recall %s, fbeta: %s" %(key_category, precision, recall, fbeta))