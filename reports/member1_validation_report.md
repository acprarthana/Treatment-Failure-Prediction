# Member 1: Data Cleaning and Validation Report

## Dataset
File: `data/interim/combined_ml_dataset_provisional.csv`

## Structural checks
| Check                        |   Value |
|:-----------------------------|--------:|
| Total rows                   |  213618 |
| Unique participants          |   12320 |
| Missing participant IDs      |       0 |
| Duplicate participant-visits |       0 |
| Missing CD4                  |   55753 |
| Missing viral load           |  107111 |
| Missing medication match     |  142498 |
| Missing outcome labels       |  213618 |

## Cohort summary
| cohort   |   lab_visit_rows |   unique_participants |   visits_with_CD4 |   visits_with_viral_load |   visits_with_medication |   visits_with_outcome |
|:---------|-----------------:|----------------------:|------------------:|-------------------------:|-------------------------:|----------------------:|
| MACS     |           143263 |                  7343 |             88459 |                    47816 |                    36333 |                     0 |
| WIHS     |            70355 |                  4977 |             69406 |                    58691 |                    34787 |                     0 |

## Important limitations
- This is a provisional combined lab-visit dataset.
- Medication information is aligned by cohort-specific participant ID and visit code.
- A matched visit does not guarantee that every medication or laboratory measurement is complete.
- Missing clinical values must be handled using a documented training-only strategy.
- Reported MACS failure-related medication-stop reasons are signals, not confirmed clinical failure outcomes.
- WIHS medication stop-reason semantics have not been verified.
- Cohort-derived sex assignments require source-document confirmation.
- Visit codes must not automatically be treated as calendar time.
- Treatment-failure outcome definition and prediction window remain unconfirmed.
- Do not train or report model performance until the target is clinically justified
  and predictors use only information available before the prediction time.

## Handoff
The dataset can be used for continued data-quality review and feature-engineering
development. Final outcome labels and modeling readiness require further validation.
