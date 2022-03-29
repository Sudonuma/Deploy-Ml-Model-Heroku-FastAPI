# Put the code for your API here.

import os
from unicodedata import name
import sys
from fastapi import FastAPI
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import train_model as tm, inference as infe, compute_model_metrics, compute_score_per_slice

import joblib
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))



if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


# Declare the data object with its components and their type.
class CensusItem(BaseModel):
    age: int = Field(example=40)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=184018)
    education: str = Field(example="Assoc-voc")
    education_num: int = Field(alias="education-num", example=11)
    marital_status: str = Field(alias="marital-status", example="Married-civ-spouse")
    occupation: str = Field(example="Machine-op-inspct")
    relationship: str = Field(example="Husband")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(alias="capital-gain", example=0)
    capital_loss: int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=38)
    native_country: str = Field(alias="native-country", example="United-States")
    


# Instantiate the app.
app = FastAPI()

@app.get("/")
async def say_hello(input_data: CensusItem):
    return {"welcome message": "Hello World!"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/")
async def inference(input_data: CensusItem):

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

    model = joblib.load("/root/Deploy-Ml-Model-Heroku-FastAPI/model/model.joblib")
    encoder = joblib.load("/root/Deploy-Ml-Model-Heroku-FastAPI/model/encoder.joblib")
    lb = joblib.load("/root/Deploy-Ml-Model-Heroku-FastAPI/model/lb.joblib")

    X_test, _ , encoder, lb= process_data(
    pd.DataFrame(input_data.dict(by_alias=True), index=[0]), categorical_features=cat_features, label=None, encoder=encoder, lb=lb, training=False
)
    preds = infe(model, X_test)
    prediction = lb.inverse_transform(preds)[0]

    return {"prediction", prediction}
