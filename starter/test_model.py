import joblib
import pandas as pd
from starter.ml.model import inference
from starter.ml.data import process_data


def test_check_result_less_than_50K():
    data = pd.DataFrame([{
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native-country": "United-States"
    }])
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    model = joblib.load("./model/RandomForestRegressor_model.pkl")
    enc = joblib.load("./model/encoder.enc")
    lb_enc = joblib.load("./model/lb.enc")

    processed_data, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        training=False,
        encoder=enc,
        lb=lb_enc)
    result = inference(model, processed_data)
    assert result == 0


def test_check_result_more_than_50K():
    data = pd.DataFrame([{
        "age": 38,
        "workclass": "Private",
        "fnlgt": 28887,
        "education": "11th",
        "education_num": 7,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native-country": "United-States"
    }])
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    model = joblib.load("./model/RandomForestRegressor_model.pkl")
    enc = joblib.load("./model/encoder.enc")
    lb_enc = joblib.load("./model/lb.enc")

    processed_data, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        training=False,
        encoder=enc,
        lb=lb_enc)
    result = inference(model, processed_data)
    assert result == 1
