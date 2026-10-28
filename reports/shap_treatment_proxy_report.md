# Gradient Boosting SHAP Explainability Report

## Model

This report explains the Gradient Boosting treatment-outcome proxy model. Gradient Boosting was selected as the best-performing model among the evaluated models based on test PR-AUC.

## Dataset

- Labelled rows: 2,377
- Unique participants: 1,192
- Training rows: 1,872
- Test rows: 505
- Transformed features: 57
- Split: participant-level holdout
- Random state: 42

## Original input features

- `CD4`
- `viral_load`
- `year`
- `CD4_previous`
- `viral_load_previous`
- `CD4_change`
- `viral_load_change`
- `years_since_previous_visit`
- `sex`
- `VISIT`

## Top SHAP features

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | numeric__year | 0.386627 |
| 2 | numeric__viral_load_previous | 0.292200 |
| 3 | numeric__CD4_previous | 0.282820 |
| 4 | numeric__viral_load_change | 0.159327 |
| 5 | numeric__CD4 | 0.098090 |
| 6 | numeric__viral_load | 0.046465 |
| 7 | categorical__VISIT_270 | 0.022374 |
| 8 | categorical__VISIT_180 | 0.017622 |
| 9 | categorical__VISIT_220 | 0.009755 |
| 10 | categorical__VISIT_320 | 0.008987 |

## Interpretation

Mean absolute SHAP values quantify how strongly each transformed feature contributed to the model's predictions on the test set. Larger values indicate greater average contribution to the model output; they do not by themselves establish clinical causality or treatment effectiveness.

## Important limitation

The target is a treatment-outcome proxy based on the available labelled data and is not independently confirmed clinical treatment failure. SHAP describes the behaviour of the trained predictive model and should not be interpreted as evidence of causal relationships.
