
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "hiv_future_virologic_prediction.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "target_future_virologic_status"
GROUP_COLUMN = "participant_id"

FEATURE_SETS = {
    "A_baseline": {
        "numeric": ["index_CD4", "index_viral_load"],
        "categorical": ["cohort"],
    },
    "B_plus_year": {
        "numeric": ["index_CD4", "index_viral_load", "index_year"],
        "categorical": ["cohort"],
    },
    "C_plus_medication_records": {
        "numeric": [
            "index_CD4", "index_viral_load",
            "index_year", "medication_records",
        ],
        "categorical": ["cohort"],
    },
}

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required = {
    TARGET, GROUP_COLUMN, "index_CD4", "index_viral_load",
    "index_year", "medication_records", "cohort",
}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=[TARGET, GROUP_COLUMN]).copy()
df[TARGET] = df[TARGET].astype(int)

if set(df[TARGET].unique()) != {0, 1}:
    raise ValueError("Target must contain classes 0 and 1.")

print(f"Rows: {len(df):,}")
print(f"Participants: {df[GROUP_COLUMN].nunique():,}")
print("Target distribution:")
print(df[TARGET].value_counts().sort_index().to_string())

# One shared participant-level split for every feature set.
splitter = GroupShuffleSplit(
    n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
train_idx, test_idx = next(
    splitter.split(df, df[TARGET], groups=df[GROUP_COLUMN])
)
train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

overlap = set(train_df[GROUP_COLUMN]) & set(test_df[GROUP_COLUMN])
if overlap:
    raise RuntimeError("Participant leakage detected.")

print(
    f"\nTrain: {len(train_df):,} rows / "
    f"{train_df[GROUP_COLUMN].nunique():,} participants"
)
print(
    f"Test: {len(test_df):,} rows / "
    f"{test_df[GROUP_COLUMN].nunique():,} participants"
)
print(f"Participant overlap: {len(overlap)}")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", min_samples_leaf=5,
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "SVM (Linear)": LinearSVC(
        class_weight="balanced", random_state=RANDOM_STATE,
        max_iter=10000,
    ),
}

try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=-1,
    )
except ImportError:
    print("XGBoost unavailable; skipping.")

try:
    from lightgbm import LGBMClassifier
    models["LightGBM"] = LGBMClassifier(
        n_estimators=200, learning_rate=0.05,
        class_weight="balanced", random_state=RANDOM_STATE,
        verbosity=-1, n_jobs=-1,
    )
except ImportError:
    print("LightGBM unavailable; skipping.")


def make_pipeline(numeric_features, categorical_features, classifier):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", clone(classifier)),
    ])


def score_values(pipeline, X):
    classifier = pipeline.named_steps["model"]
    transformed = pipeline.named_steps["preprocessor"].transform(X)
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(transformed)[:, 1]
    return classifier.decision_function(transformed)


def metrics_for(y_true, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Sensitivity": recall_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "ROC_AUC": roc_auc_score(y_true, scores),
        "PR_AUC": average_precision_score(y_true, scores),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


cv_rows = []
test_rows = []

for set_name, feature_config in FEATURE_SETS.items():
    numeric = feature_config["numeric"]
    categorical = feature_config["categorical"]
    features = numeric + categorical

    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]
    groups_train = train_df[GROUP_COLUMN]

    cv = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE
    )

    print(f"\n{'=' * 65}\nFEATURE SET: {set_name}")
    print(f"Predictors: {features}")

    for model_name, classifier in models.items():
        print(f"  Evaluating {model_name}...")
        pipeline = make_pipeline(numeric, categorical, classifier)

        try:
            fold_scores = cross_val_score(
                pipeline, X_train, y_train,
                groups=groups_train, cv=cv,
                scoring="roc_auc", n_jobs=1,
                error_score="raise",
            )
            cv_rows.append({
                "Feature_Set": set_name,
                "Model": model_name,
                "CV_ROC_AUC_Mean": fold_scores.mean(),
                "CV_ROC_AUC_Std": fold_scores.std(),
                "CV_Folds": len(fold_scores),
            })
        except Exception as error:
            print(f"    CV failed: {error}")
            cv_rows.append({
                "Feature_Set": set_name,
                "Model": model_name,
                "CV_ROC_AUC_Mean": np.nan,
                "CV_ROC_AUC_Std": np.nan,
                "CV_Folds": 0,
            })

        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            scores = score_values(pipeline, X_test)
            result = metrics_for(y_test, predictions, scores)
            test_rows.append({
                "Feature_Set": set_name,
                "Model": model_name,
                "Predictor_Count": len(features),
                **result,
            })
            print(
                f"    Test ROC-AUC={result['ROC_AUC']:.4f}, "
                f"PR-AUC={result['PR_AUC']:.4f}, "
                f"Sensitivity={result['Sensitivity']:.4f}, "
                f"Specificity={result['Specificity']:.4f}"
            )
        except Exception as error:
            print(f"    Test evaluation failed: {error}")

cv_df = pd.DataFrame(cv_rows)
test_df = pd.DataFrame(test_rows)

cv_path = REPORTS_DIR / "feature_sets_cross_validation.csv"
test_path = REPORTS_DIR / "feature_sets_test_metrics.csv"
cv_df.to_csv(cv_path, index=False)
test_df.to_csv(test_path, index=False)

print("\nCROSS-VALIDATION RESULTS")
if not cv_df.empty:
    print(
        cv_df.sort_values(
            ["Model", "Feature_Set"]
        ).round(4).to_string(index=False)
    )

print("\nHELD-OUT TEST RESULTS")
if not test_df.empty:
    print(
        test_df.sort_values(
            ["Model", "Feature_Set"]
        )[
            [
                "Feature_Set", "Model", "Predictor_Count",
                "Accuracy", "Precision", "Sensitivity",
                "Specificity", "ROC_AUC", "PR_AUC",
            ]
        ].round(4).to_string(index=False)
    )

print("\nReports saved:")
print(cv_path)
print(test_path)
print(
    "\nCaution: medication_records counts rows matched by CASEID and VISIT; "
    "it is not confirmed to represent distinct medications or disease severity. "
    "Results are exploratory and do not establish clinical validity."
)
