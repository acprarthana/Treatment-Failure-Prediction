# Member 3 Contribution Report

## Project
**Early Prediction of HIV Treatment Failure Using Longitudinal Clinical Data**

## Role and Responsibilities
Member 3 contributed to the backend prediction service, SHAP-based explainability, API input validation, and software testing.

## 1. Backend API
Implemented a FastAPI service that loads the saved Gradient Boosting model and provides:

- `GET /health` — checks API and model status.
- `POST /predict` — returns a prediction and class probabilities.
- `POST /explanation` — returns a prediction and SHAP feature contributions.

The API includes input validation, structured responses, and error handling.

## 2. Explainable AI (SHAP)
Integrated SHAP TreeExplainer to explain individual model predictions and identify influential features. The service returns signed and absolute SHAP values.

Positive SHAP values increase the model's raw score for class 1, while negative values decrease it. These values describe model behaviour and do not establish causation. SHAP values in this implementation are not direct probability changes.

Global feature-importance results and test-example explanations are stored in the `reports/` directory. The SHAP methodology and interpretation limitations are documented in `reports/member3_shap_methodology.md`.

## 3. Input Validation and Consistency
Added API validation for required and numeric inputs, consistency checks for CD4 and viral-load changes, and checks that prediction and explanation endpoints return consistent predictions.

The SHAP additivity test checks that the expected value plus the SHAP contributions reconstructs the model's class 1 raw score within numerical tolerance for the tested input.

## 4. Testing Results
The recorded automated API test run reported:

- **10 tests passed**
- **7 warnings**
- No test failures

Coverage includes health, prediction, explanation, missing and invalid inputs, request consistency, SHAP score reconstruction, and longitudinal change validation.

Detailed output: `reports/member3_api_test_report.txt`

The warnings were recorded during the test run; they did not cause test failures.

## 5. Key Files
- `backend/app/main.py` — FastAPI implementation.
- `backend/README.md` — API setup and endpoint documentation.
- `tests/test_api.py` — automated API tests.
- `src/shap_treatment_proxy.py` — SHAP analysis workflow.
- `src/save_final_model.py` — final model artifact creation.
- `reports/shap_feature_importance.csv` — global feature-importance values.
- `reports/shap_feature_importance.png` — feature-importance visualization.
- `reports/shap_treatment_proxy_report.md` — SHAP analysis report.
- `reports/member3_shap_methodology.md` — SHAP methodology, interpretation, validation, and limitations.
- `reports/member3_api_test_report.txt` — recorded API test results.

## 6. Limitations and Responsible Use
The model predicts the project's treatment-outcome proxy label. This target is not independently confirmed clinical HIV treatment failure. Results must not be presented as a clinical diagnosis or used as a substitute for clinical judgment.

SHAP explanations describe how features contribute to model outputs; they do not show that a feature causes an outcome. Correlated and derived longitudinal features may share or redistribute attribution. Independent clinical validation is required before clinical use.

Confidential datasets must remain out of version control.

## 7. Summary
Member 3's work provides a documented prediction API, SHAP-based explanations, input validation, and automated tests supporting software-level consistency and reliability. These checks do not by themselves establish clinical validity, fairness, or generalizability.