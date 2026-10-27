# Expanded-Feature Model Comparison

## Objective
Evaluate whether expanding the predictor set beyond the original three features improves prediction of `target_future_virologic_status`.

## Predictors
Six predictors were used:
- `index_CD4`
- `index_viral_load`
- `cohort`
- `index_year`
- `lab_records`
- `medication_records`

Participant identifiers were used for grouped splitting, not as predictors. Future target information and `year_gap_to_target` were excluded. `sex` was excluded because it duplicates `cohort` in this dataset.

## Evaluation setup
- Dataset rows: 36,970
- Unique participants: 5,194
- Training rows: 29,498
- Test rows: 7,472
- Training participants: 4,155
- Test participants: 1,039
- Participant overlap: 0
- Test size: 20%
- Random state: 42
- Cross-validation: 3-fold StratifiedGroupKFold on training participants

## Expanded-model test results

| Model | Accuracy | Precision | Sensitivity | Specificity | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.8860 | 0.8788 | 0.9389 | 0.8071 | 0.9448 | 0.9608 |
| LightGBM | 0.8834 | 0.8915 | 0.9168 | 0.8337 | 0.9437 | 0.9600 |
| Random Forest | 0.8778 | 0.8908 | 0.9070 | 0.8344 | 0.9410 | 0.9565 |
| Decision Tree | 0.8442 | 0.8890 | 0.8452 | 0.8427 | 0.9041 | 0.9061 |
| Logistic Regression | 0.8298 | 0.8779 | 0.8311 | 0.8277 | 0.9003 | 0.9216 |
| Linear SVM | 0.8286 | 0.8760 | 0.8311 | 0.8247 | 0.8989 | 0.9205 |

## Comparison with the earlier three-feature XGBoost baseline

| Metric | Three-feature baseline | Expanded model | Difference |
|---|---:|---:|---:|
| Accuracy | 0.8808 | 0.8860 | +0.0052 |
| ROC-AUC | 0.9197 | 0.9448 | +0.0251 |
| PR-AUC | 0.9267 | 0.9608 | +0.0341 |
| Sensitivity | 0.9385 | 0.9389 | +0.0004 |
| Specificity | 0.7947 | 0.8071 | +0.0124 |

The baseline values are from the earlier recorded evaluation; expanded-model values are from the latest run. Both scripts use the same random seed, test fraction, dataset, and participant-grouped split procedure.

## Interpretation
XGBoost achieved the highest test ROC-AUC (0.9448) and PR-AUC (0.9608) among the six expanded-feature models. Compared with the earlier XGBoost baseline, the expanded model's reported accuracy, ROC-AUC, PR-AUC, sensitivity, and specificity were all higher. The largest absolute changes were in ROC-AUC (+0.0251) and PR-AUC (+0.0341).

LightGBM had higher precision and specificity than expanded XGBoost, so model selection should consider the costs of false positives and false negatives rather than ranking by one metric alone.

## Limitations
- This is an evaluation of the dataset's defined future virologic-status target, not proof of clinically validated treatment-failure prediction.
- `medication_records` is a record-count feature and must not be interpreted as adherence or treatment effectiveness.
- The improvement is an observed result from this split; it does not by itself establish statistical significance or external generalizability.
- Confirm source-variable definitions and evaluate performance across cohorts before making clinical claims.
- The model comparison does not establish that each added feature independently caused the improvement.

## Output files
- `reports/expanded_models_cross_validation.csv`
- `reports/expanded_models_test_metrics.csv`
- `reports/expanded_models_confusion_matrices.csv`
