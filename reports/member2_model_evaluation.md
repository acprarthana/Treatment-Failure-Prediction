# Member 2 — Model Evaluation Report

## 1. Objective

Evaluate machine-learning models for predicting the dataset's `target_future_virologic_status` using longitudinal clinical features available at the index visit. Compare baseline models, boosted models, cohort-specific performance, tuned models, and a soft-voting hybrid.

The target represents future virologic status in the dataset and should not automatically be interpreted as clinically confirmed ART treatment failure.

## 2. Dataset and evaluation design

- Total records: 36,970
- Unique participants: 5,194
- MACS records: 21,138
- WIHS records: 15,832
- Features used: `index_CD4`, `index_viral_load`, and `cohort`
- Target: `target_future_virologic_status`
- Split: 80% training and 20% held-out testing using `GroupShuffleSplit`
- Grouping variable: `participant_id`
- Training records: 29,498
- Testing records: 7,472
- Participant overlap between training and test sets: 0

Numeric missing values were imputed using the median. The cohort feature was imputed using the most frequent category when necessary and one-hot encoded. Preprocessing was fitted within each model pipeline.

## 3. Initial model comparison

Six models were evaluated: Logistic Regression, Decision Tree, Random Forest, Linear SVM, XGBoost, and LightGBM.

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Accuracy |
|---|---:|---:|---:|---:|
| XGBoost | 0.9132 | 0.9197 | 0.9267 | 0.8808 |
| LightGBM | 0.9122 | 0.9182 | 0.9240 | 0.8796 |
| Random Forest | 0.9010 | 0.9087 | 0.9048 | 0.8706 |
| Decision Tree | 0.8800 | 0.8907 | 0.8868 | 0.8235 |
| Logistic Regression | 0.8757 | 0.8828 | 0.8894 | 0.7298 |
| Linear SVM | 0.8744 | 0.8822 | 0.8904 | 0.7147 |

The boosted tree models achieved the strongest overall ranking performance in the initial comparison.

## 4. Cohort-wise evaluation

Performance varied between MACS and WIHS.

In the initial comparison, XGBoost achieved:

- MACS ROC-AUC: 0.9078
- MACS sensitivity: 0.8849
- MACS specificity: 0.8773
- WIHS ROC-AUC: 0.8116
- WIHS sensitivity: 0.9734
- WIHS specificity: 0.3152

WIHS sensitivity was high, but specificity was considerably lower. Therefore, overall accuracy alone does not adequately describe performance across cohorts.

## 5. Hyperparameter tuning

XGBoost and LightGBM were tuned with randomized search using three-fold participant-grouped cross-validation and ROC-AUC as the selection metric. Twelve parameter combinations were evaluated per model.

| Model | Best CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Accuracy |
|---|---:|---:|---:|---:|
| XGBoost | 0.9132 | 0.9184 | 0.9227 | 0.8814 |
| LightGBM | 0.9134 | 0.9196 | 0.9260 | 0.8801 |

The tuned models performed similarly. LightGBM had slightly higher held-out ROC-AUC and PR-AUC, while XGBoost had slightly higher accuracy.

## 6. Hybrid model

A soft-voting ensemble was evaluated by averaging predicted probabilities from the tuned XGBoost and LightGBM models.

| Metric | XGBoost | LightGBM | Hybrid |
|---|---:|---:|---:|
| Accuracy | 0.8814 | 0.8801 | 0.8806 |
| ROC-AUC | 0.9184 | 0.9196 | 0.9191 |
| PR-AUC | 0.9227 | 0.9260 | 0.9257 |
| Sensitivity | 0.9376 | 0.9372 | 0.9385 |
| Specificity | 0.7977 | 0.7951 | 0.7944 |

The hybrid provided a very small sensitivity increase but did not outperform the best individual model across the other metrics. Its added complexity is not clearly justified by these results.

## 7. Conclusion

XGBoost and LightGBM were the strongest individual models in these experiments. Their overall results were close, and the hybrid did not provide a consistent improvement.

A key limitation is the cohort difference: WIHS had substantially lower specificity than MACS. Further work should examine threshold selection, calibration, cohort-specific error patterns, and validation on independent participants or datasets.

These results are exploratory. The target is future virologic status, and the experiments do not establish clinical readiness or confirm clinical treatment failure.

## 8. Generated artifacts

- `reports/all_models_cross_validation.csv`
- `reports/all_models_test_metrics.csv`
- `reports/all_models_confusion_matrices.csv`
- `reports/cohort_model_metrics.csv`
- `reports/cohort_model_cross_validation.csv`
- `reports/cohort_model_confusion_matrices.csv`
- `reports/tuned_boosted_model_metrics.csv`
- `reports/tuned_boosted_model_cv.csv`
- `reports/hybrid_model_metrics.csv`

## 9. Reproducibility notes

The scripts use a fixed random seed and participant-level train/test splitting. Cross-validation was grouped by participant for the tuning process. The held-out test results should be interpreted as experimental comparisons rather than definitive evidence of generalization to other populations.
