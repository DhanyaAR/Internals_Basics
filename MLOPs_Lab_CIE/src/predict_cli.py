import argparse
import joblib
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--artist_reputation_score", type=float, required=True)
parser.add_argument("--artwork_age_years", type=float, required=True)
parser.add_argument("--medium_type_index", type=float, required=True)
parser.add_argument("--exhibition_count", type=float, required=True)

args = parser.parse_args()

# Load model
model = joblib.load("models/model.pkl")

features = np.array([[ 
    args.artist_reputation_score,
    args.artwork_age_years,
    args.medium_type_index,
    args.exhibition_count
]])

prediction = model.predict(features)

print(float(prediction[0]))