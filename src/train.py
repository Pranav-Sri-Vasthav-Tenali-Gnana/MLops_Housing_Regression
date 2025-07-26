import os
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sklearn_model.joblib")

def load_data():
    print("[INFO] Fetching California Housing dataset...")
    data = fetch_california_housing()
    return data.data, data.target

def train_model(X_train, y_train):
    print("[INFO] Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f"[INFO] R² Score: {score:.4f}")
    return score

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    main()
