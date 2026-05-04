import pandas as pd
import numpy as np
import json
import os

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

def evaluate_and_log(model, name):
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X_train, y_train)
        
        # Predict & Metrics
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # 1. LOG ALL HYPERPARAMETERS (Crucial for the "All params" condition)
        mlflow.log_params(model.get_params())
        
        # 2. LOG METRICS
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        
        # 3. SET TAG
        mlflow.set_tag("experiment_type", "baseline_comparison")

        return {
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

# Execute Runs
results.append(evaluate_and_log(Ridge(), "Ridge"))
results.append(evaluate_and_log(GradientBoostingRegressor(random_state=42), "GradientBoosting"))

# Select best based on RMSE (Lower is better)
best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "gallerypulse-auction-price-lakhs",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

# Save output
os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)