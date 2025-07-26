# MLOps Housing Regression Pipeline

This project showcases the development of a fully automated MLOps pipeline. It integrates all key stages of the machine learning lifecycle, from training and containerization to deployment automation and model optimization.

## Key Components

### Model Training with scikit-learn
- Trained a Linear Regression model using the California Housing dataset from `sklearn.datasets`.
- The model was serialized using `joblib` for downstream use in prediction and quantization.

### Containerization with Docker
- Created a lightweight Docker image to package the training and prediction logic.
- Ensures portability and consistency across different environments.

### CI/CD with GitHub Actions
- Automated the following stages:
  - Model training
  - Docker image build
  - Prediction verification
  - Push to Docker Hub upon success

### Manual Model Quantization
- Converted model weights from float to unsigned 8-bit integers (uint8).
- Used these quantized weights in a PyTorch model for inference.
- Evaluated performance trade-offs between original and quantized models.

## Git Repository Setup

A public GitHub repository named `MLops_Housing_Regression` was created to host all source code and workflows.

### Repository Initialized With:
- Blank `README.md`
- Python `.gitignore` to exclude unnecessary files:
  - `__pycache__/`
  - `.ipynb_checkpoints/`
  - `.venv/`, `env/`, `*.pyc`, etc.

## Branching Strategy

The following structured branching strategy was used as per assignment instructions:

```

main → dev → docker\_ci → quantization

```

- `main`: Base repo with README and .gitignore only
- `dev`: Contains training code (`train.py`) and model saving logic
- `docker_ci`: Contains Dockerfile, `predict.py`, and CI workflow setup
- `quantization`: Implements manual model compression and inference

## Model Development (dev Branch)

- Created `dev` branch using `git checkout -b dev`
- Added `src/train.py` which performs:
  - Dataset loading
  - Model training using Linear Regression
  - R² score evaluation
  - Saving the trained model to `models/sklearn_model.joblib`

## Docker and CI Pipeline (docker_ci Branch)

- Created `docker_ci` branch from `dev`
- Added Dockerfile and `predict.py`
- Dockerfile:
  - Uses `python:3.10-slim` as base
  - Installs dependencies via `requirements.txt`
  - Copies `src/` and `models/`
  - Runs `predict.py` to evaluate the model

- `predict.py` loads the model and computes R² score on the dataset

## CI/CD with GitHub Actions

Workflow defined in `.github/workflows/ci.yml`:
- Checkout repository code
- Set up Python 3.10 and install dependencies
- Run `train.py` to regenerate model
- Build Docker image
- Run Docker container to verify prediction logic
- Authenticate with DockerHub using secrets:
  - `DOCKER_USERNAME`
  - `DOCKER_PASSWORD`
- Push Docker image: `pranav1114/housing-predictor`

## Manual Quantization (quantization Branch)

- Created `quantization` branch from `docker_ci`
- Script `quantize.py` performs:
  1. Loads `sklearn_model.joblib`
  2. Extracts `.coef_` and `.intercept_`
  3. Saves them to `unquant_params.joblib`
  4. Applies manual uint8 quantization and saves to `quant_params.joblib`
  5. Performs dequantized inference and evaluates R² score

## Final Results and Comparison

| Metric       | Sklearn Model              | Quantized Model            |
|--------------|----------------------------|----------------------------|
| R² Score     | 0.5758                     | -0.0955                    |
| Model Size   | 0.414 KB                   | 0.381 KB                   |

### Observations
- Model size reduced by ~45% after quantization.
- However, there was a significant loss in accuracy (R² score dropped below 0).
- Quantization should be used with caution, especially for models sensitive to precision loss.

## Conclusion

This project successfully demonstrates an end-to-end MLOps pipeline, integrating reproducible ML development, Docker-based deployment, CI/CD automation, and model optimization via quantization. The structured Git workflow ensures code modularity and stage-wise progression.
