import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split, KFold, ParameterSampler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import mlflow

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("auction_price_lakhs", axis=1)
y = df["auction_price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ GIVEN PARAM GRID (MUST MATCH QUESTION)
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5]
}

# Random sampling (10 trials)
param_list = list(ParameterSampler(param_grid, n_iter=10, random_state=42))

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mlflow.set_experiment("gallerypulse-auction-price-lakhs")

best_rmse = float("inf")
best_params = None
final_best_mae = None
final_best_cv_mae = None # Track this for the JSON

with mlflow.start_run(run_name="tuning-gallerypulse"):

    for params in param_list:
        with mlflow.start_run(nested=True, run_name="Trial"):
            model = GradientBoostingRegressor(**params, random_state=42)
            cv_rmses = []
            cv_maes = [] # Added to satisfy JSON requirement

            for train_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)

                cv_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
                cv_maes.append(mean_absolute_error(y_val, preds))

            avg_cv_rmse = np.mean(cv_rmses)
            avg_cv_mae = np.mean(cv_maes)

            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
            test_mae = mean_absolute_error(y_test, test_preds)

            mlflow.log_params(params)
            mlflow.log_metrics({
                "cv_rmse": avg_cv_rmse,
                "cv_mae": avg_cv_mae,
                "test_rmse": test_rmse,
                "test_mae": test_mae
            })

            # Logic: Select best by RMSE
            if avg_cv_rmse < best_rmse:
                best_rmse = avg_cv_rmse
                best_params = params
                final_best_mae = test_mae
                final_best_cv_mae = avg_cv_mae 

# Final Output Construction
output = {
    "search_type": "random",
    "n_folds": 5,
    "total_trials": len(param_list),
    "best_params": best_params,
    "best_mae": final_best_mae,
    "best_cv_mae": final_best_cv_mae, # Now correctly holds a MAE value
    "parent_run_name": "tuning-gallerypulse"
}

os.makedirs("results", exist_ok=True)
with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)