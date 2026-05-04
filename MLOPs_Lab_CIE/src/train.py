import pandas as pd
import numpy as np
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("auction_price_lakhs", axis=1)
y = df["auction_price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("gallerypulse-auction-price-lakhs")

results = []

# Function to train, evaluate, log, and return model
def evaluate_and_log(model, name):
    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # Log ALL hyperparameters
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

        # Tag
        mlflow.set_tag("experiment_type", "baseline_comparison")

        return {
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "model": model   # keep model for saving later
        }

# Run both models
ridge_result = evaluate_and_log(Ridge(), "Ridge")
gb_result = evaluate_and_log(
    GradientBoostingRegressor(random_state=42),
    "GradientBoosting"
)

results.append(ridge_result)
results.append(gb_result)

# Select best model by RMSE
best = min(results, key=lambda x: x["rmse"])

# Save best model
os.makedirs("models", exist_ok=True)

if best["name"] == "Ridge":
    joblib.dump(ridge_result["model"], "models/model.pkl")
else:
    joblib.dump(gb_result["model"], "models/model.pkl")

# Remove model objects before saving JSON
for r in results:
    r.pop("model")

# Prepare output JSON
output = {
    "experiment_name": "gallerypulse-auction-price-lakhs",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

# Save results
os.makedirs("results", exist_ok=True)

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Step 1 completed: Model trained, MLflow logged, model saved.")