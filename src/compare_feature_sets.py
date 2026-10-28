
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

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

# Compare feature sets while keeping the same participant-level split.
FEATURE_SETS = {
    "A_baseline": {
        "numeric": ["index_CD4", "index_viral_load"],
        "categorical": ["cohort"],
    },
    "B_plus_year": {
        "numeric": [
            "index_CD4",
            "index_viral_load",
            "index_year",
        ],
        "categorical": ["cohort"],
    },
    "C_plus_medication_records": {
        "numeric": [
            "index_CD4",
            "index_viral_load",
            "index_year",
            "medication_records",
        ],
        "categorical": ["cohort"],
    },
    "D_expanded_seven_features": {
        "numeric": [
            "index_CD4",
            "index_viral_load",
            "index_year",
            "medication_records",
            "lab_records",
        ],
        "categorical": ["cohort", "sex"],
    },
}

MODELS = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "DecisionTree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "LinearSVM": LinearSVC(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
    ),
}


def make_pipeline(numeric_features, categorical_features, model):
    """Build preprocessing and model pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def get_scores(model, X):
    """Return continuous scores for ROC-AUC and PR-AUC."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return model.predict(X)


def calculate_metrics(y_true, y_pred, y_score):
    """Calculate test-set classification metrics."""

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "sensitivity": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "specificity": specificity,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Validate the dataset columns after loading the data.
    required = {
        TARGET,
        GROUP_COLUMN,
        "index_CD4",
        "index_viral_load",
        "index_year",
        "medication_records",
        "lab_records",
        "cohort",
        "sex",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Remove rows without target or participant identifier.
    df = df.dropna(
        subset=[TARGET, GROUP_COLUMN]
    ).copy()

    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="coerce"
    )
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(
            "Target must contain only binary labels 0 and 1."
        )

    # Hold out entire participants, preventing participant overlap.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            df[TARGET],
            groups=df[GROUP_COLUMN],
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    overlap = set(train_df[GROUP_COLUMN]) & set(
        test_df[GROUP_COLUMN]
    )

    if overlap:
        raise RuntimeError(
            "Participant leakage detected between train and test."
        )

    print("\nDATASET SUMMARY")
    print("-" * 50)
    print(f"Rows: {len(df):,}")
    print(f"Participants: {df[GROUP_COLUMN].nunique():,}")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print(f"Participant overlap: {len(overlap)}")
    print("\nTarget counts:")
    print(df[TARGET].value_counts().sort_index())

    model_results = []
    cv_results = []

    for feature_set_name, feature_config in FEATURE_SETS.items():
        numeric_features = feature_config["numeric"]
        categorical_features = feature_config["categorical"]

        features = numeric_features + categorical_features

        X_train = train_df[features]
        y_train = train_df[TARGET]
        X_test = test_df[features]
        y_test = test_df[TARGET]

        print(f"\n\nFEATURE SET: {feature_set_name}")
        print(f"Number of predictors: {len(features)}")
        print(f"Predictors: {features}")

        # Cross-validation also keeps each participant in one fold.
        n_splits = min(
            5,
            train_df[GROUP_COLUMN].nunique(),
        )

        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        for model_name, model in MODELS.items():
            pipeline = make_pipeline(
                numeric_features,
                categorical_features,
                model,
            )

            # Fit only on training participants.
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_score = get_scores(pipeline, X_test)

            metrics = calculate_metrics(
                y_test,
                y_pred,
                y_score,
            )

            model_results.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "predictor_count": len(features),
                    **metrics,
                }
            )

            try:
                scores = cross_val_score(
                    make_pipeline(
                        numeric_features,
                        categorical_features,
                        model,
                    ),
                    X_train,
                    y_train,
                    groups=train_df[GROUP_COLUMN],
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                )

                cv_results.append(
                    {
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "cv_roc_auc_mean": scores.mean(),
                        "cv_roc_auc_std": scores.std(),
                    }
                )

                cv_text = (
                    f"{scores.mean():.4f} "
                    f"+/- {scores.std():.4f}"
                )

            except Exception as exc:
                cv_text = f"CV failed: {exc}"

            print(f"\nModel: {model_name}")
            print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"Test PR-AUC: {metrics['pr_auc']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"Specificity: {metrics['specificity']:.4f}")
            print(f"CV ROC-AUC: {cv_text}")

    results_df = pd.DataFrame(model_results)
    cv_df = pd.DataFrame(cv_results)

    test_path = REPORTS_DIR / "feature_sets_test_metrics.csv"
    cv_path = REPORTS_DIR / "feature_sets_cross_validation.csv"

    results_df.to_csv(test_path, index=False)
    cv_df.to_csv(cv_path, index=False)

    print("\n\nTEST METRICS COMPARISON")
    print("-" * 80)
    print(
        results_df[
            [
                "feature_set",
                "model",
                "predictor_count",
                "roc_auc",
                "pr_auc",
                "sensitivity",
                "specificity",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nCROSS-VALIDATION ROC-AUC")
    print("-" * 80)
    if not cv_df.empty:
        print(cv_df.round(4).to_string(index=False))

    print("\nSaved reports:")
    print(test_path)
    print(cv_path)

    print(
        "\nLIMITATIONS:\n"
        "- The target is future virologic status, not confirmed "
        "clinical treatment failure.\n"
        "- The 200 copies/mL threshold is exploratory and needs "
        "source/clinical validation.\n"
        "- Participant-level splitting prevents the same "
        "participant appearing in both train and test sets.\n"
        "- Random participant splitting does not establish "
        "performance on future calendar periods.\n"
        "- lab_records and medication_records are counts; their "
        "meaning and availability at prediction time must be checked.\n"
        "- Results are exploratory and are not clinical advice."
    )


if __name__ == "__main__":
    main()
