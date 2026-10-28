\# Treatment Failure Prediction — Model Evaluation Report



\*\*Project:\*\* Treatment-Failure-Prediction  

\*\*Report file:\*\* `reports/treatment\_proxy\_evaluation\_report.md`  

\*\*Evaluation status:\*\* Preliminary model evaluation  

\*\*Target:\*\* MACS medication stop-reason proxy  

\*\*Recommended model based on currently available results:\*\* Gradient Boosting



\---



\## 1. Objective



The objective of this evaluation is to compare machine-learning models for identifying records associated with a reported treatment-failure-related medication stop reason.



The current target is derived from MACS medication stop-reason fields (`STOT1`, `STOT2`, `STOT3`, and `STOTH`). It is a proxy outcome and must not be interpreted as clinically confirmed antiretroviral treatment failure.



The evaluation is preliminary because the labeled dataset is limited, the positive class is relatively rare, and the prediction horizon has not been clinically defined.



\## 2. Dataset and Target Definition



The enriched dataset is:



`data/interim/hiv\_longitudinal\_outcomes\_enriched\_final\_candidate.csv`



| Dataset characteristic | Value |

|---|---:|

| Total rows | 213,618 |

| MACS rows | 143,263 |

| WIHS rows | 70,355 |

| Labeled treatment-outcome records | 2,377 |

| Positive proxy labels | 262 |

| Negative proxy labels | 2,115 |

| Unlabeled/unknown treatment outcomes | 211,241 |



\### Target-label interpretation



\- \*\*Positive (1):\*\* A code 65 treatment-failure-related stop reason was recorded.

\- \*\*Negative (0):\*\* At least one interpretable stop reason was recorded, but none was code 65.

\- \*\*Unknown:\*\* No interpretable stop reason was available, or the source's outcome mapping was not verified.



A negative proxy label does not establish successful treatment. Likewise, a positive proxy label is not equivalent to an independently confirmed clinical treatment-failure diagnosis.



The WIHS outcome mapping remains unverified, so WIHS records have not been assigned this treatment-outcome target.



\## 3. Data Preparation and Features



The model pipeline uses the following ten features:



1\. `CD4`

2\. `viral\_load`

3\. `year`

4\. `CD4\_previous`

5\. `viral\_load\_previous`

6\. `CD4\_change`

7\. `viral\_load\_change`

8\. `years\_since\_previous\_visit`

9\. `sex`

10\. `VISIT`



Previous measurements and change features are calculated using participant-level visit histories ordered by participant, year, and visit.



The existing pipeline applies median imputation and scaling to numeric features, and mode imputation and one-hot encoding to categorical features. The data split uses `GroupShuffleSplit` with a 20% test proportion and `random\_state=42`, grouping by participant.



\*\*Important validation requirement:\*\* All longitudinal feature calculations and preprocessing must be checked to ensure that no future information leaks into a prediction. For prospective prediction, each feature must be available at the intended prediction time. Preprocessing parameters and decision thresholds must be learned using training/validation data only.



\## 4. Models Compared



The project includes seven candidate models:



\- Logistic Regression

\- Decision Tree

\- Random Forest

\- Linear Support Vector Machine

\- Gradient Boosting

\- Extra Trees

\- XGBoost



The complete seven-model comparison should be reviewed in:



`reports/treatment\_proxy\_7\_model\_comparison.csv`



Threshold-dependent results should be reviewed in:



`reports/treatment\_proxy\_threshold\_results.csv`



The results reproduced below cover Gradient Boosting, Random Forest, and XGBoost. They do not establish the ranking of all seven models.



\## 5. Evaluation Metrics



Because positive proxy labels are uncommon, accuracy alone is not sufficient. The evaluation considers:



\- \*\*ROC-AUC:\*\* Ability to rank positive cases above negative cases across thresholds.

\- \*\*PR-AUC:\*\* Precision-recall performance, especially relevant with an imbalanced target.

\- \*\*Precision:\*\* Proportion of predicted positives that are actual positive proxy labels.

\- \*\*Recall/Sensitivity:\*\* Proportion of positive proxy labels identified by the model.

\- \*\*Specificity:\*\* Proportion of negative proxy labels correctly identified.

\- \*\*F1-score:\*\* Harmonic mean of precision and recall.



The decision threshold affects precision, recall, specificity, and F1-score. Therefore, threshold-dependent metrics must be interpreted alongside the threshold used.



\## 6. Results



\### 6.1 Model comparison using the currently available results



| Metric | Gradient Boosting | Random Forest | XGBoost |

|---|---:|---:|---:|

| Selected threshold | 0.15 | 0.35 | 0.47 |

| ROC-AUC | \*\*0.7478\*\* | 0.7301 | 0.7125 |

| PR-AUC | \*\*0.2092\*\* | 0.2077 | 0.1980 |

| Precision | \*\*0.2114\*\* | 0.1745 | 0.1755 |

| Recall/Sensitivity | 0.6981 | \*\*0.7736\*\* | 0.6226 |

| Specificity | \*\*0.6947\*\* | 0.5708 | 0.6571 |

| F1-score | \*\*0.3246\*\* | 0.2847 | 0.2739 |



\*Values are from the existing evaluation report. Metrics are rounded to four decimal places.\*



\### 6.2 Gradient Boosting confusion matrix



At the selected threshold of 0.15:



| Outcome | Count |

|---|---:|

| True negatives (TN) | 314 |

| False positives (FP) | 138 |

| False negatives (FN) | 16 |

| True positives (TP) | 37 |



Gradient Boosting identifies 37 of the 53 positive proxy labels in the test set, corresponding to a recall of approximately 69.8%. It predicts 175 cases as positive, of which 37 are true positive proxy labels, giving precision of approximately 21.1%.



These results demonstrate the trade-off between detecting more positive proxy cases and generating false-positive predictions.



\### 6.3 Random Forest confusion matrix



At the selected threshold of 0.35:



| Outcome | Count |

|---|---:|

| True negatives (TN) | 258 |

| False positives (FP) | 194 |

| False negatives (FN) | 12 |

| True positives (TP) | 41 |



Random Forest has the highest recall among these three models, at approximately 77.4%. However, it also has more false positives and lower specificity and precision than Gradient Boosting.



\### 6.4 XGBoost confusion matrix



At the selected threshold of 0.47:



| Outcome | Count |

|---|---:|

| True negatives (TN) | 297 |

| False positives (FP) | 155 |

| False negatives (FN) | 20 |

| True positives (TP) | 33 |



XGBoost has lower ROC-AUC, PR-AUC, precision, recall, specificity, and F1-score than Gradient Boosting in the reported results.



\## 7. Interpretation and Model Recommendation



Based on the available metrics for these three models, \*\*Gradient Boosting is the strongest overall candidate\*\*. It has the highest ROC-AUC, PR-AUC, precision, specificity, and F1-score in this comparison.



Random Forest may be preferable when identifying a larger proportion of positive proxy cases is the primary goal, because it has the highest recall of the three models. This comes with a higher false-positive count.



XGBoost is not the best-performing model in the currently available comparison. However, a final claim that Gradient Boosting is the best of all seven models must wait until `treatment\_proxy\_7\_model\_comparison.csv` has been checked.



The threshold of 0.15 for Gradient Boosting is a decision threshold, not a universal clinical cutoff. It should be selected according to the project's intended use and validated on appropriate held-out data.



\## 8. Limitations



1\. \*\*Proxy target:\*\* The label is derived from medication stop reasons, not independently confirmed clinical treatment failure.

2\. \*\*Limited labeled sample:\*\* Only 2,377 records have known proxy labels, including 262 positive records.

3\. \*\*Class imbalance:\*\* Positive proxy labels represent a small minority of labeled records.

4\. \*\*Unknown outcomes:\*\* Most dataset rows do not have an interpretable treatment-outcome label.

5\. \*\*Unverified WIHS mapping:\*\* WIHS records are not assigned treatment-outcome labels because comparability of the source coding has not been established.

6\. \*\*Prediction horizon undefined:\*\* The target has not yet been defined as an outcome occurring within a specified future interval.

7\. \*\*Participant dependence:\*\* Multiple visits from the same participant are correlated. Participant-grouped splitting helps avoid direct participant overlap between training and test sets, but repeated evaluation and external validation are still needed.

8\. \*\*Potential temporal leakage:\*\* Longitudinal feature engineering must be verified to ensure that only information available at prediction time is used.

9\. \*\*Threshold dependence:\*\* Precision, recall, specificity, and F1-score depend on the selected threshold.

10\. \*\*Clinical applicability:\*\* These preliminary results do not demonstrate clinical utility, prospective performance, or suitability for patient-care decisions.



\## 9. Data Validation Notes



The existing merge-validation report records 213,618 rows before and after enrichment, 12,320 unique participants, and zero duplicate participant-visit rows in the enriched dataset.



One discrepancy needs reconciliation before final sign-off: the merge-validation report records 314 code-65-positive medication participant-visits, while the enriched dataset contains 262 positive `treatment\_outcome` labels. These figures may reflect different merge stages or populations, but the difference should be explained by tracing the source records and merge logic.



Relevant file:



`reports/outcome\_merge\_validation.txt`



The final report should not describe the discrepancy as resolved until the underlying records have been checked.



\## 10. Recommended Next Steps



1\. Inspect `reports/treatment\_proxy\_7\_model\_comparison.csv` and confirm the complete seven-model ranking.

2\. Reconcile the 314 code-65-positive medication participant-visits with the 262 positive outcome labels.

3\. Verify the target-label logic against the source codebook and representative records.

4\. Confirm that all longitudinal features use only information available at the intended prediction time.

5\. Evaluate performance using repeated participant-grouped validation or grouped cross-validation.

6\. Select the decision threshold using validation data and a clearly stated objective.

7\. Report uncertainty, including confidence intervals where feasible, and assess calibration.

8\. Assess subgroup performance and possible cohort-specific differences where sample sizes permit.

9\. Define the prediction horizon and obtain a clinically validated target before describing the model as a treatment-failure predictor.

10\. Keep the current work labeled as \*\*preliminary proxy-outcome modeling\*\*, not a validated clinical prediction system.



\## 11. Conclusion



The current evaluation indicates that Gradient Boosting outperforms Random Forest and XGBoost on most of the reported metrics, while Random Forest achieves higher recall among those three models. Gradient Boosting is therefore the recommended provisional choice based on the available comparison.



This recommendation is limited to the results currently reported and must be checked against all seven candidate models. Because the outcome is a medication stop-reason proxy, most records are unlabeled, the prediction horizon is undefined, and a label-count discrepancy remains unresolved, the model should not yet be presented as a clinically validated treatment-failure prediction system.



\---



\*\*Existing supporting reports\*\*



\- `reports/treatment\_proxy\_7\_model\_comparison.csv`

\- `reports/treatment\_proxy\_threshold\_results.csv`

\- `reports/treatment\_proxy\_final\_report.md`

\- `reports/outcome\_merge\_validation.txt`



