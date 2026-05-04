import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load datasets
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

# Combine datasets
combined_df = pd.concat([train_df, new_df], ignore_index=True)

# Original test set (from 25 rows) - for fair comparison
X_orig = train_df.drop("auction_price_lakhs", axis=1)
y_orig = train_df["auction_price_lakhs"]
_, X_test_orig, _, y_test_orig = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42
)

# Combined training data (everything except original test set)
X_train_comb = combined_df.drop("auction_price_lakhs", axis=1)
y_train_comb = combined_df["auction_price_lakhs"]

# Champion model - load saved model from Task 1
champion_model = joblib.load("models/model.pkl")
champion_preds = champion_model.predict(X_test_orig)
champion_rmse = np.sqrt(mean_squared_error(y_test_orig, champion_preds))

# Retrained model - same type (Ridge), trained on combined data
retrained_model = Ridge()
retrained_model.fit(X_train_comb, y_train_comb)
retrained_preds = retrained_model.predict(X_test_orig)
retrained_rmse = np.sqrt(mean_squared_error(y_test_orig, retrained_preds))

# Compare and decide
improvement = champion_rmse - retrained_rmse
action = "promoted" if improvement >= 1.0 else "kept_champion"

# Output JSON
output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_rmse": champion_rmse,
    "retrained_rmse": retrained_rmse,
    "improvement": improvement,
    "min_improvement_threshold": 1.0,
    "action": action,
    "comparison_metric": "rmse"
}

os.makedirs("results", exist_ok=True)
with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Step 4 completed successfully")