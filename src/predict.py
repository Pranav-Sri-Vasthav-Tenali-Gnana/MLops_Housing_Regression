# src/predict.py

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
import os

def main():
    model_path = "models/sklearn_model.joblib"

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        return

    print("[INFO] Loading model...")
    model = joblib.load(model_path)

    print("[INFO] Fetching dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target

    print("[INFO] Running predictions...")
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)

    print(f"[PREDICT] R² score on full dataset: {score:.4f}")

if __name__ == "__main__":
    main()
