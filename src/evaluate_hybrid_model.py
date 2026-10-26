
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "interim" / "hiv_future_virologic_prediction.csv"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["index_CD4", "index_viral_load", "cohort"]
TARGET = "target_future_virologic_status"
GROUP = "participant_id"
RANDOM_STATE = 42
THRESHOLD = 0.5


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


def get_metrics(y_true, probabilities, threshold=THRESHOLD):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Rows": len(y_true),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Sensitivity": recall_score(
            y_true, predictions, zero_division=0
        ),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
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
    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    overlap = set(train_groups) & set(test_groups)

    print(f"Dataset rows: {len(df):,}")
    print(f"Training rows: {len(train_idx):,}")
    print(f"Testing rows: {len(test_idx):,}")
    print(f"Participant overlap: {len(overlap)}")

    if overlap:
        raise RuntimeError("Participant overlap detected!")

    preprocessor = make_preprocessor()

    xgb = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
            subsample=0.8,
            n_estimators=150,
            min_child_weight=5,
            max_depth=3,
            learning_rate=0.03,
            colsample_bytree=0.8,
        )),
    ])

    lgbm = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", LGBMClassifier(
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=-1,
            subsample=0.8,
            num_leaves=15,
            n_estimators=150,
            min_child_samples=20,
            max_depth=7,
            learning_rate=0.05,
            colsample_bytree=1.0,
        )),
    ])

    print("\nFitting XGBoost...")
    xgb.fit(X_train, y_train)

    print("Fitting LightGBM...")
    lgbm.fit(X_train, y_train)

    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    lgbm_prob = lgbm.predict_proba(X_test)[:, 1]

    # Equal-weight soft voting: average predicted probabilities.
    hybrid_prob = (xgb_prob + lgbm_prob) / 2

    models = {
        "Tuned XGBoost": xgb_prob,
        "Tuned LightGBM": lgbm_prob,
        "Hybrid Soft Voting": hybrid_prob,
    }

    all_results = []

    for model_name, probabilities in models.items():
        overall = get_metrics(y_test, probabilities)
        overall["Model"] = model_name
        overall["Cohort"] = "ALL"
        all_results.append(overall)

        for cohort in sorted(X_test["cohort"].dropna().unique()):
            mask = X_test["cohort"].eq(cohort).to_numpy()
            cohort_result = get_metrics(
                y_test.iloc[np.flatnonzero(mask)],
                probabilities[mask],
            )
            cohort_result["Model"] = model_name
            cohort_result["Cohort"] = cohort
            all_results.append(cohort_result)

    results = pd.DataFrame(all_results)
    results = results[
        ["Model", "Cohort", "Rows", "Accuracy", "Precision",
         "Sensitivity", "Specificity", "ROC_AUC", "PR_AUC",
         "TN", "FP", "FN", "TP"]
    ]

    results = results.sort_values(
        ["Cohort", "ROC_AUC"], ascending=[True, False]
    )

    output_path = REPORTS_DIR / "hybrid_model_metrics.csv"
    results.to_csv(output_path, index=False)

    print("\nHYBRID VS INDIVIDUAL MODELS")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
