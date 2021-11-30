# Script to train machine learning model.
import joblib
# import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model
from ml.model import inference
import pandas as pd

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('../data/census.csv')
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
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, training=False, encoder = encoder, lb = lb
)

# Train and save a model.
model = train_model(X_train,y_train)
joblib.dump(model,"../model/RandomForestRegressor_model.pkl")
joblib.dump(encoder, "../model/encoder.enc")
joblib.dump(lb, "../model/lb.enc")
# mlflow.sklearn.save_model(model,"../model/model")