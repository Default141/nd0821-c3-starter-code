# Put the code for your API here.
"""
main inferance for model API
Author: Virut Ratinimittham
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data

app = FastAPI()


class Data(BaseModel):
    """Body for data

    Args:
        BaseModel ([type]): data structure
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: float
    native_country: str
    class Config:
        schema_extra = {
            "example": {
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
        }

@app.get("/")
def read_root():
    """Greeting message"""
    return {"Welcome to Salary prediction model"}


@app.post("/model/inferance")
def inferance_model(data_input: Data):
    """
    inferance model

    input: data_input data for prediction

    output: prediction result
    """
    data = pd.DataFrame([{
            "age": data_input.age,
            "workclass": data_input.workclass,
            "fnlgt": data_input.fnlgt,
            "education": data_input.education,
            "education-num": data_input.education_num,
            "marital-status": data_input.marital_status,
            "occupation": data_input.occupation,
            "relationship": data_input.relationship,
            "race": data_input.race,
            "sex": data_input.sex,
            "capital-gain": data_input.capital_gain,
            "capital-loss": data_input.capital_loss,
            "hours-per-week": data_input.hours_per_week,
            "native-country": data_input.native_country
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

    model = joblib.load("starter/model/RandomForestRegressor_model.pkl")
    enc = joblib.load("starter/model/encoder.enc")
    lb_enc = joblib.load("starter/model/lb.enc")

    processed_data, _, _, _ = process_data(
        data, categorical_features=cat_features,
        training=False,
        encoder=enc,
        lb=lb_enc)
    result = inference(model, processed_data)
    if result[0] >= 0.1:
        return '>50K'
    return '<=50K'
