# Script to train machine learning model.
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from starter.ml.data import process_data
import starter.ml.model

# Add code to load in the data.
def read_return_train_test_data(path):

    data = pd.read_csv(path)
    train, test = train_test_split(data, test_size=0.20)

    return train, test


def train_model(cat_features, train):
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label='salary', training=True
)

    # Proces the test data with the process_data function.

    # Train and save a model.
    trained_model = starter.ml.model.train_model(X_train, y_train)
    # save model
    joblib.dump(trained_model, f"/root/Deploy-Ml-Model-Heroku-FastAPI/model/model.joblib")
    joblib.dump(encoder, f"/root/Deploy-Ml-Model-Heroku-FastAPI/model/encoder.joblib")
    joblib.dump(lb, f"/root/Deploy-Ml-Model-Heroku-FastAPI/model/lb.joblib")


def evaluate_model(cat_features, test):

    model = joblib.load("/root/Deploy-Ml-Model-Heroku-FastAPI/model/model.joblib")
    encoder = joblib.load("/root/Deploy-Ml-Model-Heroku-FastAPI/model/encoder.joblib")
    lb = joblib.load("/root/Deploy-Ml-Model-Heroku-FastAPI/model/lb.joblib")

    X_test, y_test, encoder, lb= process_data(
    test, categorical_features=cat_features, label='salary', encoder=encoder, lb=lb, training=False
)
    preds = starter.ml.model.inference(model, X_test)
    metrics = starter.ml.model.compute_model_metrics(y_test, preds)
    prediction = lb.inverse_transform(preds)

    # Compute score per slice
    starter.ml.model.compute_score_per_slice(model, test, encoder, lb, cat_features)

    return metrics, prediction


if __name__ == "__main__":

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

    # Load and split data
    train, test = read_return_train_test_data(f"/root/Deploy-Ml-Model-Heroku-FastAPI/data/clean_data.csv")
    
    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label='salary', training=True
    )
    X_test, y_test, encoder, lb= process_data(
        test, categorical_features=cat_features, label='salary', encoder=encoder, lb=lb, training=False
    )

    # Train and save a model.
    train_model(cat_features, train)

    # Evaluate model and compute score per slice
    metrics, prediction = evaluate_model(cat_features, test)

    