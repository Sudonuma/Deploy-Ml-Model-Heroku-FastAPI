from urllib import request
from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_api_welcome():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"welcome message": "Hello World!"}

def test_high_predictions():
    r = client.post("/", json={"age": 35,
        "workclass": "Private",
        "fnlgt": 220098,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Other-service",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"})
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}

def test_low_predictions():
    r = client.post("/", json={"age": 50,
        "workclass": "Private",
        "fnlgt": 176609,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Divorced",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"})

    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}