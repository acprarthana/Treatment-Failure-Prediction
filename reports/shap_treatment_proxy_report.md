# SHAP Explainability — Treatment Outcome Proxy

## Dataset and model

- Labelled rows: 2,377
- Training rows: 1,872
- Test rows: 505
- Training participants: 953
- Test participants: 239
- Model: XGBoost
- Explainer: SHAP TreeExplainer
- Split: participant-level 80/20 holdout

## Input variables

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

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | numeric__year | 0.788242 |
| 2 | numeric__viral_load_previous | 0.514685 |
| 3 | numeric__CD4_previous | 0.492640 |
| 4 | numeric__viral_load | 0.253156 |
| 5 | numeric__CD4 | 0.186505 |
| 6 | numeric__viral_load_change | 0.169500 |
| 7 | numeric__CD4_change | 0.143659 |
| 8 | categorical__VISIT_180 | 0.053859 |
| 9 | numeric__years_since_previous_visit | 0.037643 |
| 10 | categorical__VISIT_500 | 0.030354 |

## Important interpretation note

SHAP values explain the XGBoost model's predictions. They do not establish causal relationships or clinical treatment failure. The underlying target is a treatment-outcome proxy as documented by the existing evaluation pipeline.
