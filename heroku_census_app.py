import requests


input_data = {"age": 35,
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
        "native-country": "United-States"
    }

r = requests.post('https://census-salary-prediction.herokuapp.com/', json=input_data)

# assert r.status_code == 200

print(r)
print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())