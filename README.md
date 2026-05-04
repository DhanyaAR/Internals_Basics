# Internals_Basics
# GalleryPulse Auction Price Prediction — MLOps Lab CIE

An end-to-end MLOps pipeline to predict art auction prices for GalleryPulse, an art auction house.


## Dataset Features
| Feature | Description | Range |
|---|---|---|
| artist_reputation_score | Artist reputation | 1–10 |
| artwork_age_years | Age of artwork | 1–200 |
| medium_type_index | Medium type | 1–5 |
| exhibition_count | Number of exhibitions | 0–20 |
| auction_price_lakhs | Target variable | — |

## Tasks

### Task 1 — Experiment Tracking & Model Comparison
- Trained Ridge and GradientBoosting models
- Logged hyperparameters, MAE, RMSE, R2 to MLflow
- Experiment name: `gallerypulse-auction-price-lakhs`
- Best model selected by RMSE: **Ridge**
- Results: `results/step1_s1.json`

### Task 2 — Hyperparameter Tuning
- Tuned GradientBoosting with random search
- 5-fold cross-validation, 10 trials
- Nested MLflow runs under parent `tuning-gallerypulse`
- Results: `results/step2_s2.json`

### Task 3 — Docker Packaging
- Containerized CLI prediction tool
- Base image: `python:3.10-slim`
- Image: `gallerypulse-predictor:v1`
- Results: `results/step3_s3.json`

### Task 4 — Retraining Pipeline
- Combined training + new data (25 + 20 = 45 rows)
- Retrained Ridge on combined data
- Champion RMSE: 9.084 → Retrained RMSE: 6.238
- Improvement: 2.845 (≥ 1.0 threshold) → **Promoted**
- Results: `results/step4_s8.json`

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Task 1 — Train & Compare Models
```bash
python src/train.py
```

### Task 2 — Hyperparameter Tuning
```bash
python src/tune.py
```

### Task 3 — Docker CLI Predictor
```bash
docker build -t gallerypulse-predictor:v1 .
docker run gallerypulse-predictor:v1 \
  --artist_reputation_score 4.5 \
  --artwork_age_years 89 \
  --medium_type_index 3 \
  --exhibition_count 11
```

### Task 4 — Retraining Pipeline
```bash
python src/retrain.py
```

### View MLflow UI
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser.

## Results Summary
| Task | Output File | Key Result |
|---|---|---|
| Task 1 | step1_s1.json | Best model: Ridge (RMSE: 9.084) |
| Task 2 | step2_s2.json | Best params: n_estimators=50, lr=0.05, depth=3 |
| Task 3 | step3_s3.json | Prediction: 52.28 lakhs |
| Task 4 | step4_s8.json | Action: Promoted (improvement: 2.845) |

## Tech Stack
- Python 3.10
- scikit-learn
- MLflow
- Docker
- pandas, numpy
