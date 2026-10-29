# Member 3: SHAP Methodology and Limitations

## 1. Method

SHAP (SHapley Additive exPlanations) assigns feature contributions to individual model predictions relative to a baseline output. This project uses `shap.TreeExplainer` with the trained Gradient Boosting classifier.

SHAP explains the behaviour of the fitted model. It does not establish causal relationships or prove that changing a clinical measurement would change a patient's outcome.

## 2. Interpreting API explanations

For the current binary Gradient Boosting model, the API reports SHAP contributions to the class 1 raw model score.

- **Positive contribution:** moves the explained raw score upward.
- **Negative contribution:** moves the explained raw score downward.
- **Absolute SHAP value:** indicates the magnitude of a feature's contribution in the units of the explained output.
- **Baseline plus contributions:** should reconstruct the explained raw score, subject to numerical precision and the explainer's additivity behaviour.

SHAP values in this API are not direct probability changes. The model's predicted probability must be read separately from the prediction response.

## 3. Global and individual explanations

Global feature importance is calculated using mean absolute SHAP values over the evaluated test examples. It summarizes the magnitude of contributions across those examples.

An individual explanation describes contributions for one input and may have a different feature ranking or contribution direction from the global summary.

Neither global importance nor an individual explanation establishes clinical importance or causation.

## 4. Assumptions and limitations

- **Model dependence:** explanations depend on the trained model, preprocessing, input features, and selected output.
- **Feature dependence:** longitudinal variables such as current and previous CD4 counts, viral loads, and changes are related. Attribution can depend on how feature dependence is handled by the explainer.
- **Correlated and derived variables:** related features may share or redistribute attribution; a ranking is not a definitive measure of independent importance.
- **Data dependence:** explanations reflect the available labelled data and evaluation population.
- **Target limitation:** the target is a treatment-outcome proxy, not independently confirmed clinical HIV treatment failure.
- **No causal interpretation:** SHAP values do not show that a feature caused a treatment outcome.
- **No standalone clinical use:** the explanations are intended for research and model inspection, not standalone diagnosis or treatment decisions.

## 5. Validation

The API test suite includes a check that the explainer's baseline plus SHAP contributions reconstructs the class 1 raw model score for a test input within numerical tolerance.

This verifies additivity for that tested input. It does not establish clinical validity, fairness, or the reliability of every explanation.

## 6. Responsible reporting

Reports and demonstrations should distinguish model predictions from confirmed clinical outcomes, describe SHAP values in raw-score units for this implementation, and avoid causal claims. Any future model or explainer change should trigger a review of the explanation method and its validation tests.
