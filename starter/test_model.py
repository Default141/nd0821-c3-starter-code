import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from starter.ml.model import train_model
from starter.ml.data import process_data


data = pd.read_csv('./data/census.csv')
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
    train, categorical_features=cat_features, label="salary", training=True
)
print(type(X_train))
model = train_model(X_train, y_train)


def test_train_model():
    assert model is not None


def test_X_train_data_type():
    assert isinstance(X_train, np.ndarray)


def test_y_train_data_type():
    assert isinstance(y_train, np.ndarray)
