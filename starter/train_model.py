# Script to train machine learning model.
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from ml.data import process_data
import ml.model

# Add code to load in the data.
data = pd.read_csv(f"/root/Deploy-Ml-Model-Heroku-FastAPI/data/clean_data.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label='salary', training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
trained_model = ml.model.train_model(X_train, y_train)
# save model
joblib.dump(trained_model, f"/root/Deploy-Ml-Model-Heroku-FastAPI/model/model.joblib")
joblib.dump(encoder, f"/root/Deploy-Ml-Model-Heroku-FastAPI/model/encoder.joblib")
joblib.dump(lb, f"/root/Deploy-Ml-Model-Heroku-FastAPI/model/lb.joblib")


# TODO change all of this with onehot encoding use lb to binarize Y

# y = test['salary']
# yy = np.where(y=='<=50K', 0, 1)

# X_test = test.drop(['salary'], axis=1)
X_test, y_test, encoder, lb= process_data(
    test, categorical_features=cat_features, label='salary', encoder=encoder, lb=lb, training=False
)
model = joblib.load("/root/Deploy-Ml-Model-Heroku-FastAPI/model/model.joblib")
encoder = joblib.load("/root/Deploy-Ml-Model-Heroku-FastAPI/model/encoder.joblib")
lb = joblib.load("/root/Deploy-Ml-Model-Heroku-FastAPI/model/lb.joblib")

# inference
preds = ml.model.inference(model, X_test)
metrics = ml.model.compute_model_metrics(y_test, preds)
prediction = lb.inverse_transform(preds)

ml.model.compute_score_per_slice(model, test, encoder, lb, cat_features)