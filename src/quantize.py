# src/quantize.py

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

def quantize_to_uint8(array, scale=0.01):
    """Quantize a float32 array to uint8 with a given scale."""
    quantized = np.clip(np.round(array / scale), 0, 255).astype(np.uint8)
    return quantized, scale

def dequantize_from_uint8(quantized_array, scale=0.01):
    """Dequantize a uint8 array back to float32 using the provided scale."""
    return quantized_array.astype(np.float32) * scale

def main():
    # Load the trained scikit-learn model
    model = joblib.load("models/sklearn_model.joblib")
    coef = model.coef_
    intercept = model.intercept_

    # Save unquantized parameters
    unquantized_parameters = {
        "coef": coef,
        "intercept": intercept
    }
    joblib.dump(unquantized_parameters, "models/unquant_params.joblib")

    # Quantize coefficients and intercept
    coef_q, coef_scale = quantize_to_uint8(coef)
    intercept_q, intercept_scale = quantize_to_uint8(np.array([intercept]))

    # Save quantized parameters
    quantized_parameters = {
        "coef_q": coef_q,
        "intercept_q": intercept_q,
        "coef_scale": coef_scale,
        "intercept_scale": intercept_scale
    }
    joblib.dump(quantized_parameters, "models/quant_params.joblib")

    # Load dataset for evaluation
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Dequantize parameters for prediction
    coef_dq = dequantize_from_uint8(coef_q, coef_scale)
    intercept_dq = dequantize_from_uint8(intercept_q, intercept_scale)[0]

    # Predict using manually dequantized weights
    y_pred = np.dot(X, coef_dq) + intercept_dq
    r2 = r2_score(y, y_pred)

    print(f"R² score using manually dequantized weights: {r2:.4f}")

if __name__ == "__main__":
    main()
