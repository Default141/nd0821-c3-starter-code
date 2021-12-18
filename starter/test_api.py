from fastapi import FastAPI
import requests
import json

app = FastAPI()


def test_check_response_code_get():
    response = requests.get("http://127.0.0.1:8000/")
    assert response.status_code == 200


def test_check_response_content_get():
    response = requests.get("http://127.0.0.1:8000/")
    assert response.json()[0] == "Welcome to Salary prediction model"


def test_check_response_code_post():
    data = {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    data = json.dumps(data).encode('utf8')
    response = requests.post("http://127.0.0.1:8000/model/inferance", data)
    assert response.status_code == 200


def test_check_response_content_post_1():
    data = {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    data = json.dumps(data).encode('utf8')
    response = requests.post("http://127.0.0.1:8000/model/inferance", data)
    assert response.json() == '<=50K'


def test_check_response_content_post_2():
    data = {
            "age": 38,
            "workclass": "Private",
            "fnlgt": 28887,
            "education": "11th",
            "education_num": 7,
            "marital_status": "Married-civ-spouse",
            "occupation": "Sales",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 50,
            "native_country": "United-States"
            }
    data = json.dumps(data).encode('utf8')
    response = requests.post("http://127.0.0.1:8000/model/inferance", data)
    assert response.json() == '>50K'
