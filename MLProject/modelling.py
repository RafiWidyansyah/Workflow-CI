import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os


tracking_uri = "http://127.0.0.1:5000/"
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    local_mlflow_dir = os.path.join(os.path.dirname(__file__), "mlruns")
    mlflow.set_tracking_uri(f"file://{os.path.abspath(local_mlflow_dir)}")

mlflow.set_experiment("Used Car Price Prediction")

# Get the directory of this script and construct the CSV path
csv_path = os.path.join(os.path.dirname(__file__), "used_car_price_dataset_cleaned_scaled.csv")
data = pd.read_csv(csv_path)
 
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("selling_price", axis=1),
    data["selling_price"],
    random_state=42,
    test_size=0.2
)

input_example = X_train[0:5]

with mlflow.start_run():
        mlflow.autolog()
    # Train model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
        )   
    # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

