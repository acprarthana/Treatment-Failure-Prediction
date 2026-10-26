
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "interim" / "hiv_future_virologic_prediction.csv"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_ITER = 12
N_SPLITS = 3

FEATURES = ["index_CD4", "index_viral_load", "cohort"]
TARGET = "target_future_virologic_status"
GROUP = "participant_id"


def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric, ["index_CD4", "index_viral_load"]),
        ("categorical", categorical, ["cohort"]),
    ])


def get_models_and_search_spaces():
    return {
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
            {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [2, 3, 4, 5],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__min_child_weight": [1, 3, 5],
            },
        ),
        "LightGBM": (
            LGBMClassifier(
                objective="binary",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1,
            ),
            {
                "model__n_estimators": [150, 250, 350],
                "model__max_depth": [-1, 3, 5, 7],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__num_leaves": [15, 31, 50],
                "model__min_child_samples": [10, 20, 40],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        ),
    }


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "Rows": len(y_true),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Sensitivity": recall_score(
            y_true, predictions, zero_division=0
        ),
        "Specificity": specificity,
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "PR_AUC": average_precision_score(y_true, probabilities),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET, GROUP]).copy()

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    groups = df[GROUP]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    print(f"Rows: {len(df):,}")
    print(f"Training rows: {len(train_idx):,}")
    print(f"Testing rows: {len(test_idx):,}")
    print(
        "Participant overlap:",
        len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])),
    )

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []
    cv_results = []
    best_models = {}

    for name, (estimator, search_space) in get_models_and_search_spaces().items():
        print(f"\nTuning {name}...")

        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ])

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=search_space,
            n_iter=N_ITER,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            verbose=1,
            error_score="raise",
        )

        search.fit(X_train, y_train, groups=groups_train)

        best_models[name] = search.best_estimator_

        print(f"Best grouped CV ROC-AUC: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

        cv_results.append({
            "Model": name,
            "Best_CV_ROC_AUC": search.best_score_,
            "Best_Parameters": str(search.best_params_),
        })

        probabilities = search.best_estimator_.predict_proba(X_test)[:, 1]

        overall = calculate_metrics(y_test, probabilities)
        overall["Model"] = name
        overall["Cohort"] = "ALL"
        results.append(overall)

        for cohort in sorted(X_test["cohort"].dropna().unique()):
            mask = X_test["cohort"].eq(cohort).to_numpy()
            cohort_metrics = calculate_metrics(
                y_test.iloc[np.flatnonzero(mask)],
                probabilities[mask],
            )
            cohort_metrics["Model"] = name
            cohort_metrics["Cohort"] = cohort
            results.append(cohort_metrics)

            print(
                f"{cohort}: ROC-AUC={cohort_metrics['ROC_AUC']:.4f}, "
                f"Sensitivity={cohort_metrics['Sensitivity']:.4f}, "
                f"Specificity={cohort_metrics['Specificity']:.4f}"
            )

    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df[
        ["Model", "Cohort", "Rows", "Accuracy", "Precision",
         "Sensitivity", "Specificity", "ROC_AUC", "PR_AUC",
         "TN", "FP", "FN", "TP"]
    ]
    metrics_df = metrics_df.sort_values(
        ["Cohort", "ROC_AUC"], ascending=[True, False]
    )

    metrics_path = REPORTS_DIR / "tuned_boosted_model_metrics.csv"
    cv_path = REPORTS_DIR / "tuned_boosted_model_cv.csv"

    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(cv_results).to_csv(cv_path, index=False)

    print("\nFINAL HELD-OUT TEST RESULTS")
    print(metrics_df.round(4).to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {cv_path}")


if __name__ == "__main__":
    main()
