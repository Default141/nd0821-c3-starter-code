# Script to train machine learning model.
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import inference

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('../data/census.csv')
# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
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
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

np.savetxt("../data/X_train.csv", X_train)
np.savetxt("../data/y_train.csv", y_train)

# print(type(y_test))
# f = open("test.txt", "a")
# for i in y_test:
#     f.write(np.array2string(i))
# f.close()

# Train and save a model.
model = train_model(X_train, y_train)
pred_validate = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(pred_validate, y_test)
metrics = {
    "precision": [precision],
    "recall": [recall],
    "fbeta": [fbeta]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("../data/metrics.csv")
# np.savetxt("../data/metrics.csv", metrics)
joblib.dump(model, "../model/RandomForestRegressor_model.pkl")
joblib.dump(encoder, "../model/encoder.enc")
joblib.dump(lb, "../model/lb.enc")
# mlflow.sklearn.save_model(model,"../model/model")


def model_slice_metrix():
    education_slice = ['Bachelors', 'Some-college', '11th', 'HS-grad',
                       'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                       '7th-8th', '12th', 'Masters', '1st-4th', '10th',
                       'Doctorate', '5th-6th', 'Preschool']
    print(data)
    for slice in education_slice:
        data_slice = data[data["education"] == slice]
        X_test_slice, y_test_slice, _, _ = process_data(
            data_slice,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )
        pred_validate = inference(model, X_test_slice)
        precision, recall, fbeta = compute_model_metrics(pred_validate,
                                                         y_test_slice)
        metrics = {
            "precision": [precision],
            "recall": [recall],
            "fbeta": [fbeta]
        }
        print(metrics)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(f"../data/metrics_slice_{slice}.csv")
