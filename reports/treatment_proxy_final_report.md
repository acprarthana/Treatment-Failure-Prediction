# HIV Treatment Outcome Proxy — Model Evaluation

## Dataset and task
- Labelled rows: 2,377 (from the current training run)
- Positive labels: 262
- Negative labels: 2,115
- Input features: 10 candidate features, including derived previous-visit features
- Evaluation split: participant-level holdout
- Target: medication-stop-reason proxy, not independently confirmed clinical failure

## Threshold-tuned results
| Model             |   Threshold_selected_on_validation |   Validation_F1_at_selected_threshold |   Test_ROC_AUC |   Test_PR_AUC |   Test_Precision |   Test_Sensitivity |   Test_Specificity |   Test_F1 |   TN |   FP |   FN |   TP |
|:------------------|-----------------------------------:|--------------------------------------:|---------------:|--------------:|-----------------:|-------------------:|-------------------:|----------:|-----:|-----:|-----:|-----:|
| Gradient Boosting |                               0.15 |                              0.384615 |       0.747767 |      0.209212 |         0.211429 |           0.698113 |           0.69469  |  0.324561 |  314 |  138 |   16 |   37 |
| Random Forest     |                               0.35 |                              0.392694 |       0.73013  |      0.20772  |         0.174468 |           0.773585 |           0.570796 |  0.284722 |  258 |  194 |   12 |   41 |
| XGBoost           |                               0.47 |                              0.427746 |       0.712515 |      0.198046 |         0.175532 |           0.622642 |           0.65708  |  0.273859 |  297 |  155 |   20 |   33 |

## Provisional result
Highest test PR-AUC among the three threshold-tuned models:
**Gradient Boosting** (PR-AUC 0.209).

This is a preliminary result from one holdout split. Thresholds were selected on validation data.

## Limitations
1. The target is a proxy label, not confirmed clinical treatment failure.
2. Visit ordering must be validated before interpreting previous-visit features.
3. Confirm all predictors are available before the event being predicted.
4. Report precision, sensitivity, specificity, F1 and PR-AUC; accuracy alone is insufficient.
5. Validate with repeated/grouped cross-validation before claiming generalizability.
