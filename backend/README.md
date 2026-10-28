# Treatment Failure Prediction API

## Overview

This FastAPI service provides predictions from a saved Gradient Boosting model and SHAP-based feature explanations.

The model predicts the project's treatment-outcome proxy label. Its output is a research result, not an independently confirmed clinical diagnosis.

## Requirements

- Python 3.11
- Dependencies listed in the repository-root `requirements.txt`
- The saved model file at `models/gradient_boosting_treatment_proxy.joblib`

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Start the API

From the repository root, run:

```bash
python -m uvicorn backend.app.main:app --reload
```

The local server runs at `http://127.0.0.1:8000`.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Endpoints

### GET `/health`

Checks whether the API is running and reports the model service information.

### POST `/predict`

Returns the model prediction and class probabilities.

Example request:

```json
{
  "CD4": 500,
  "viral_load": 1000,
  "year": 2018,
  "CD4_previous": 520,
  "viral_load_previous": 800,
  "CD4_change": -20,
  "viral_load_change": 200,
  "years_since_previous_visit": 1,
  "sex": "Male",
  "VISIT": 180
}
```

### POST `/explanation`

Returns a prediction and the top feature contributions computed with SHAP TreeExplainer.

Each feature entry contains its name, signed SHAP value, and absolute SHAP value. Positive and negative SHAP values indicate the direction of influence on the model output; they do not establish causation.

## Automated tests

From the repository root, run:

```bash
python -m pytest tests/test_api.py -v
```

The API test suite contains five tests covering health, prediction, explanations, missing fields, and invalid numeric input.

## SHAP artifacts

- `reports/shap_feature_importance.csv` - global feature importance values.
- `reports/shap_feature_importance.png` - global feature-importance chart.
- `reports/shap_treatment_proxy_report.md` - SHAP analysis report.
- `reports/shap_test_values.csv` - SHAP values for test examples.
- `reports/shap_test_predictions.csv` - predictions for test examples.

## Interpretation and limitations

SHAP values explain the behavior of the fitted model; they do not establish causation. The proxy target must not be interpreted as confirmed clinical treatment failure without independent clinical validation.

Model predictions and explanations are intended for research and evaluation, not as a substitute for clinical judgment.

Use appropriate access controls for sensitive clinical data. Do not commit confidential datasets to version control.