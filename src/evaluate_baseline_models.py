
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Paths and dataset
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "interim" / "hiv_future_virologic_prediction.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False)

TARGET = "target_future_virologic_status"
GROUP = "participant_id"

# --------------------------------------------------
# 2. Select predictors
# --------------------------------------------------
# Exclude IDs, visit codes, and future information.
# lab_records is constant, so it is excluded too.
features = [
    "index_CD4",
    "index_viral_load",
    "cohort",
]

X = df[features].copy()
y = df[TARGET].astype(int)
groups = df[GROUP]

# Encode cohort as a numeric indicator.
X["cohort"] = X["cohort"].map({"MACS": 0, "WIHS": 1})

# --------------------------------------------------
# 3. Split by participant: 80% train, 20% test
# --------------------------------------------------
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

# Confirm no participant appears in both sets.
assert set(groups_train).isdisjoint(set(groups_test))

print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)
print(f"Total rows: {len(df):,}")
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Training participants: {groups_train.nunique():,}")
print(f"Testing participants: {groups_test.nunique():,}")
print("Participant overlap: 0")
print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())
print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------
numeric_features = ["index_CD4", "index_viral_load"]
categorical_features = ["cohort"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", "passthrough", categorical_features),
])

# --------------------------------------------------
# 5. Define models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

# --------------------------------------------------
# 6. Train and evaluate
# --------------------------------------------------
results = []
confusion_rows = []

for model_name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    results.append(metrics)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    confusion_rows.append({
        "model": model_name,
        "true_0_pred_0": cm[0, 0],
        "true_0_pred_1": cm[0, 1],
        "true_1_pred_0": cm[1, 0],
        "true_1_pred_1": cm[1, 1],
    })

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Future status 0", "Future status 1"],
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(cm)

# --------------------------------------------------
# 7. Save evaluation reports
# --------------------------------------------------
results_df = pd.DataFrame(results)
confusion_df = pd.DataFrame(confusion_rows)

metrics_path = REPORT_DIR / "baseline_model_metrics.csv"
confusion_path = REPORT_DIR / "baseline_model_confusion_matrices.csv"

results_df.to_csv(metrics_path, index=False)
confusion_df.to_csv(confusion_path, index=False)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.round(4).to_string(index=False))
print(f"\nMetrics saved to: {metrics_path}")
print(f"Confusion matrices saved to: {confusion_path}")
print(
    "\nResearch note: the target is future virologic status, "
    "not a clinically confirmed treatment-failure diagnosis."
)

# --------------------------------------------------
# 8. Evaluate each cohort separately
# --------------------------------------------------
cohort_results = []

for model_name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    test_cohorts = df.iloc[test_idx]["cohort"].reset_index(drop=True)

    test_predictions = pipeline.predict(X_test)
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    cohort_test = test_cohorts.reset_index(drop=True)
    actual = y_test.reset_index(drop=True)

    for cohort_name in sorted(cohort_test.unique()):
        mask = cohort_test == cohort_name
        y_true = actual[mask]
        y_pred = test_predictions[mask.to_numpy()]
        y_prob = test_probabilities[mask.to_numpy()]

        cohort_results.append({
            "model": model_name,
            "cohort": cohort_name,
            "test_rows": int(mask.sum()),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(
                y_true, y_pred
            ),
            "precision": precision_score(
                y_true, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, zero_division=0
            ),
            "f1": f1_score(
                y_true, y_pred, zero_division=0
            ),
            "roc_auc": (
                roc_auc_score(y_true, y_prob)
                if y_true.nunique() == 2 else float("nan")
            ),
        })

cohort_metrics = pd.DataFrame(cohort_results)
cohort_path = REPORT_DIR / "baseline_model_metrics_by_cohort.csv"
cohort_metrics.to_csv(cohort_path, index=False)

print("\n" + "=" * 60)
print("COHORT-SPECIFIC TEST RESULTS")
print("=" * 60)
print(cohort_metrics.round(4).to_string(index=False))
print(f"\nCohort metrics saved to: {cohort_path}")


# 9. Cohort-only baseline
print("\n" + "=" * 60)
print("COHORT-ONLY BASELINE")
print("=" * 60)

cohort_only_preprocessor = ColumnTransformer(
    transformers=[
        ("cohort", OneHotEncoder(handle_unknown="ignore"), ["cohort"])
    ],
    remainder="drop"
)

cohort_only_model = Pipeline(steps=[
    ("preprocessor", cohort_only_preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

cohort_only_model.fit(X_train, y_train)
cohort_only_predictions = cohort_only_model.predict(X_test)
cohort_only_probabilities = cohort_only_model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, cohort_only_predictions), 4))
print("Balanced accuracy:", round(balanced_accuracy_score(y_test, cohort_only_predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, cohort_only_probabilities), 4))
print("\nClassification report:")
print(classification_report(
    y_test,
    cohort_only_predictions,
    target_names=["Future status 0", "Future status 1"],
    zero_division=0
))

