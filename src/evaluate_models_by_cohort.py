
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "interim" / "hiv_future_virologic_prediction.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

TARGET = "target_future_virologic_status"
GROUP = "participant_id"
NUMERIC = ["index_CD4", "index_viral_load"]
CATEGORICAL = ["cohort"]
FEATURES = NUMERIC + CATEGORICAL
SEED = 42

# --------------------------------------------------
# 1. Load and validate data
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
required = FEATURES + [TARGET, GROUP]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=[TARGET, GROUP]).copy()
df[TARGET] = df[TARGET].astype(int)

if set(df[TARGET].unique()) != {0, 1}:
    raise ValueError("Target must contain classes 0 and 1.")

X = df[FEATURES].copy()
y = df[TARGET].copy()
groups = df[GROUP].copy()

print(f"Rows: {len(df):,}")
print(f"Participants: {groups.nunique():,}")
print("\nCohort counts:")
print(df["cohort"].value_counts())

# --------------------------------------------------
# 2. Participant-level train/test split
# --------------------------------------------------

split = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED,
)

train_idx, test_idx = next(split.split(X, y, groups))

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()
groups_train = groups.iloc[train_idx].copy()
groups_test = groups.iloc[test_idx].copy()

if set(groups_train).intersection(set(groups_test)):
    raise RuntimeError("Participant leakage detected!")

test_cohorts = df.iloc[test_idx]["cohort"].reset_index(drop=True)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Participant overlap: {len(set(groups_train) & set(groups_test))}")

# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        NUMERIC,
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ]),
        CATEGORICAL,
    ),
])

# --------------------------------------------------
# 4. Define six models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        min_samples_leaf=5,
        random_state=SEED,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        max_iter=10000,
        random_state=SEED,
    ),
}

try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
except ImportError:
    print("XGBoost unavailable; skipping.")

try:
    from lightgbm import LGBMClassifier

    models["LightGBM"] = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1,
    )
except ImportError:
    print("LightGBM unavailable; skipping.")

# --------------------------------------------------
# 5. Metric calculations
# --------------------------------------------------

def calculate_metrics(actual, predicted, scores):
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    return {
        "Rows": len(actual),
        "Class_0_Count": int((actual == 0).sum()),
        "Class_1_Count": int((actual == 1).sum()),
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "Sensitivity": recall_score(
            actual, predicted, pos_label=1, zero_division=0
        ),
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ROC_AUC": roc_auc_score(actual, scores),
        "PR_AUC": average_precision_score(actual, scores),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

# --------------------------------------------------
# 6. Training-only grouped cross-validation
# --------------------------------------------------

n_splits = min(3, groups_train.nunique())
cv = StratifiedGroupKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=SEED,
)

cohort_results = []
cv_results = []
confusion_results = []
failed_models = []

for name, classifier in models.items():
    print(f"\nTraining and evaluating: {name}")

    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", clone(classifier)),
    ])

    # CV is restricted to training data.
    try:
        from sklearn.model_selection import cross_val_score

        scores_cv = cross_val_score(
            pipeline,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
            error_score="raise",
        )

        cv_results.append({
            "Model": name,
            "CV_ROC_AUC_Mean": scores_cv.mean(),
            "CV_ROC_AUC_Std": scores_cv.std(),
            "CV_Folds": len(scores_cv),
        })

        print(f"Training CV ROC-AUC: {scores_cv.mean():.4f}")

    except Exception as exc:
        print(f"CV failed for {name}: {exc}")
        cv_results.append({
            "Model": name,
            "CV_ROC_AUC_Mean": np.nan,
            "CV_ROC_AUC_Std": np.nan,
            "CV_Folds": 0,
        })

    # Fit once on the complete training set.
    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        fitted_classifier = pipeline.named_steps["model"]
        transformed_test = pipeline.named_steps[
            "preprocessor"
        ].transform(X_test)

        if hasattr(fitted_classifier, "predict_proba"):
            prediction_scores = fitted_classifier.predict_proba(
                transformed_test
            )[:, 1]
        else:
            prediction_scores = fitted_classifier.decision_function(
                transformed_test
            )

        # Overall held-out test results.
        overall = calculate_metrics(
            y_test.to_numpy(),
            predictions,
            prediction_scores,
        )

        cohort_results.append({
            "Model": name,
            "Cohort": "ALL",
            **overall,
        })

        # Evaluate the same fitted model separately in each cohort.
        for cohort_name in sorted(test_cohorts.dropna().unique()):
            mask = (
                test_cohorts.to_numpy() == cohort_name
            )

            cohort_y = y_test.to_numpy()[mask]
            cohort_pred = predictions[mask]
            cohort_scores = prediction_scores[mask]

            if len(np.unique(cohort_y)) < 2:
                print(
                    f"Skipping ROC/PR-AUC for {cohort_name}: "
                    "test subset has only one class."
                )
                continue

            metrics = calculate_metrics(
                cohort_y,
                cohort_pred,
                cohort_scores,
            )

            cohort_results.append({
                "Model": name,
                "Cohort": cohort_name,
                **metrics,
            })

            print(
                f"{cohort_name}: "
                f"Accuracy={metrics['Accuracy']:.4f}, "
                f"ROC-AUC={metrics['ROC_AUC']:.4f}, "
                f"PR-AUC={metrics['PR_AUC']:.4f}, "
                f"Sensitivity={metrics['Sensitivity']:.4f}, "
                f"Specificity={metrics['Specificity']:.4f}"
            )

        confusion_results.append({
            "Model": name,
            "Cohort": "ALL",
            **{
                key: overall[key]
                for key in ["TN", "FP", "FN", "TP"]
            },
        })

        for cohort_name in sorted(test_cohorts.dropna().unique()):
            mask = test_cohorts.to_numpy() == cohort_name
            actual = y_test.to_numpy()[mask]
            predicted = predictions[mask]

            tn, fp, fn, tp = confusion_matrix(
                actual, predicted, labels=[0, 1]
            ).ravel()

            confusion_results.append({
                "Model": name,
                "Cohort": cohort_name,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            })

    except Exception as exc:
        print(f"Evaluation failed for {name}: {exc}")
        failed_models.append({
            "Model": name,
            "Error": str(exc),
        })

# --------------------------------------------------
# 7. Save reports
# --------------------------------------------------

cohort_df = pd.DataFrame(cohort_results)
cv_df = pd.DataFrame(cv_results)
confusion_df = pd.DataFrame(confusion_results)
failed_df = pd.DataFrame(failed_models)

cohort_df.to_csv(
    REPORTS / "cohort_model_metrics.csv", index=False
)
cv_df.to_csv(
    REPORTS / "cohort_model_cross_validation.csv", index=False
)
confusion_df.to_csv(
    REPORTS / "cohort_model_confusion_matrices.csv", index=False
)

if not failed_df.empty:
    failed_df.to_csv(
        REPORTS / "cohort_model_failures.csv", index=False
    )

print("\n" + "=" * 70)
print("COHORT-WISE HELD-OUT TEST RESULTS")
print("=" * 70)

if not cohort_df.empty:
    columns = [
        "Model", "Cohort", "Rows", "Accuracy", "Precision",
        "Sensitivity", "Specificity", "ROC_AUC", "PR_AUC",
    ]

    print(
        cohort_df[columns]
        .sort_values(["Cohort", "ROC_AUC"], ascending=[True, False])
        .round(4)
        .to_string(index=False)
    )

print("\nReports saved in:", REPORTS)
print(" - cohort_model_metrics.csv")
print(" - cohort_model_cross_validation.csv")
print(" - cohort_model_confusion_matrices.csv")

if not failed_df.empty:
    print("Some models failed. Check cohort_model_failures.csv.")
