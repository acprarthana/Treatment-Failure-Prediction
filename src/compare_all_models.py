
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score,
)

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

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "hiv_future_virologic_prediction.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "target_future_virologic_status"
GROUP_COLUMN = "participant_id"

NUMERIC_FEATURES = [
    "index_CD4",
    "index_viral_load",
]

CATEGORICAL_FEATURES = [
    "cohort",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# 2. Load and validate the dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Check the dataset filename and location."
    )

df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [TARGET, GROUP_COLUMN]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.dropna(subset=[TARGET, GROUP_COLUMN]).copy()
df[TARGET] = df[TARGET].astype(int)

X = df[FEATURES].copy()
y = df[TARGET].copy()
groups = df[GROUP_COLUMN].copy()

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "The target must contain both classes, coded as 0 and 1."
    )

print("\nDataset loaded successfully.")
print(f"Rows: {len(df):,}")
print(f"Unique participants: {groups.nunique():,}")
print("\nTarget distribution:")
print(y.value_counts().sort_index())


# --------------------------------------------------
# 3. Split by participant to prevent data leakage
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()
y_test = y.iloc[test_indices].copy()

groups_train = groups.iloc[train_indices].copy()
groups_test = groups.iloc[test_indices].copy()

overlap = set(groups_train).intersection(set(groups_test))

if overlap:
    raise RuntimeError(
        "Participant leakage detected between train and test sets."
    )

print("\nParticipant-level split completed.")
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print(f"Training participants: {groups_train.nunique():,}")
print(f"Testing participants: {groups_test.nunique():,}")
print(f"Participant overlap: {len(overlap)}")


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)


# --------------------------------------------------
# 5. Define the requested models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "SVM (Linear)": LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=10000,
    ),
}

# Optional models: use them when their packages are installed.
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
except ImportError:
    print(
        "\nXGBoost skipped: its package is not installed "
        "or could not be imported."
    )

try:
    from lightgbm import LGBMClassifier

    models["LightGBM"] = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=-1,
    )
except ImportError:
    print(
        "\nLightGBM skipped: its package is not installed "
        "or could not be imported."
    )


# --------------------------------------------------
# 6. Evaluation metrics
# --------------------------------------------------

def get_scores(fitted_pipeline, features):
    """Return prediction scores for ranking metrics."""

    classifier = fitted_pipeline.named_steps["model"]
    transformed_features = (
        fitted_pipeline.named_steps["preprocessor"].transform(features)
    )

    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(transformed_features)[:, 1]

    if hasattr(classifier, "decision_function"):
        return classifier.decision_function(transformed_features)

    raise ValueError(
        "This model provides neither probabilities nor decision scores."
    )


def calculate_metrics(y_true, predictions, scores):
    """Calculate classification metrics for binary targets."""

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Sensitivity": recall_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "Specificity": specificity,
        "ROC_AUC": roc_auc_score(y_true, scores),
        "PR_AUC": average_precision_score(y_true, scores),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# --------------------------------------------------
# 7. Cross-validation on training data only
# --------------------------------------------------

number_of_splits = min(3, groups_train.nunique())

cv = StratifiedGroupKFold(
    n_splits=number_of_splits,
    shuffle=True,
    random_state=RANDOM_STATE,
)

cv_results = []
test_results = []
confusion_results = []
fitted_models = {}

for model_name, classifier in models.items():
    print(f"\nEvaluating: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", clone(classifier)),
        ]
    )

    # Cross-validation uses training data only.
    try:
        fold_scores = cross_val_score(
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
            "Model": model_name,
            "CV_ROC_AUC_Mean": fold_scores.mean(),
            "CV_ROC_AUC_Std": fold_scores.std(),
            "CV_Folds": len(fold_scores),
        })

        print(
            f"  Training CV ROC-AUC: "
            f"{fold_scores.mean():.4f} "
            f"+/- {fold_scores.std():.4f}"
        )

    except Exception as error:
        print(f"  Cross-validation failed: {error}")
        cv_results.append({
            "Model": model_name,
            "CV_ROC_AUC_Mean": np.nan,
            "CV_ROC_AUC_Std": np.nan,
            "CV_Folds": 0,
        })

    # Fit the model on all training data.
    try:
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        scores = get_scores(pipeline, X_test)

        metrics = calculate_metrics(
            y_test,
            predictions,
            scores,
        )

        test_results.append({
            "Model": model_name,
            **metrics,
        })

        confusion_results.append({
            "Model": model_name,
            **{
                key: metrics[key]
                for key in ["TN", "FP", "FN", "TP"]
            },
        })

        fitted_models[model_name] = pipeline

        print(
            f"  Test Accuracy: {metrics['Accuracy']:.4f} | "
            f"ROC-AUC: {metrics['ROC_AUC']:.4f} | "
            f"PR-AUC: {metrics['PR_AUC']:.4f} | "
            f"Sensitivity: {metrics['Sensitivity']:.4f} | "
            f"Specificity: {metrics['Specificity']:.4f}"
        )

    except Exception as error:
        print(f"  Test evaluation failed: {error}")


# --------------------------------------------------
# 8. Save evaluation reports
# --------------------------------------------------

cv_df = pd.DataFrame(cv_results)
test_df = pd.DataFrame(test_results)
confusion_df = pd.DataFrame(confusion_results)

cv_path = REPORTS_DIR / "all_models_cross_validation.csv"
test_path = REPORTS_DIR / "all_models_test_metrics.csv"
confusion_path = REPORTS_DIR / "all_models_confusion_matrices.csv"

cv_df.to_csv(cv_path, index=False)
test_df.to_csv(test_path, index=False)
confusion_df.to_csv(confusion_path, index=False)

print("\n" + "=" * 60)
print("MODEL COMPARISON: HELD-OUT TEST RESULTS")
print("=" * 60)

if not test_df.empty:
    display_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "ROC_AUC",
        "PR_AUC",
    ]

    print(
        test_df[display_columns]
        .sort_values("ROC_AUC", ascending=False)
        .round(4)
        .to_string(index=False)
    )

print("\nReports saved:")
print(cv_path)
print(test_path)
print(confusion_path)

print(
    "\nImportant: the highest test score alone does not prove "
    "that a model is the best final model. Compare cross-validation, "
    "sensitivity, specificity, ROC-AUC and PR-AUC before deciding."
)
