
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / (
    "hiv_longitudinal_outcomes_enriched_final_candidate.csv"
)
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TARGET = "treatment_outcome"
GROUP = "participant_id"

# Ten candidate predictors. No record counts or outcome-derived predictors.
NUMERIC_FEATURES = [
    "CD4",
    "viral_load",
    "year",
    "CD4_previous",
    "viral_load_previous",
    "CD4_change",
    "viral_load_change",
    "years_since_previous_visit",
]
CATEGORICAL_FEATURES = ["sex", "VISIT"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA, low_memory=False)

    required = [
        TARGET, GROUP, "CD4", "viral_load", "year", "sex", "VISIT"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    # Create chronological visit ordering before filtering to labelled rows.
    df["_year_order"] = pd.to_numeric(df["year"], errors="coerce")
    df["_visit_order"] = pd.to_numeric(df["VISIT"], errors="coerce")
    df = df.sort_values(
        [GROUP, "_year_order", "_visit_order"],
        na_position="last"
    ).copy()

    # Convert measurements to numeric.
    for col in ["CD4", "viral_load", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Historical features are calculated within each participant only.
    grouped = df.groupby(GROUP, sort=False)
    df["CD4_previous"] = grouped["CD4"].shift(1)
    df["viral_load_previous"] = grouped["viral_load"].shift(1)
    df["_previous_year"] = grouped["year"].shift(1)

    df["CD4_change"] = df["CD4"] - df["CD4_previous"]
    df["viral_load_change"] = (
        df["viral_load"] - df["viral_load_previous"]
    )
    df["years_since_previous_visit"] = (
        df["year"] - df["_previous_year"]
    )

    # Keep only rows with a known binary target and a participant ID.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    data = df.loc[
        df[TARGET].isin([0, 1]) & df[GROUP].notna()
    ].copy()
    data[TARGET] = data[TARGET].astype(int)

    # Check target labels and feature availability.
    if data[TARGET].nunique() != 2:
        raise ValueError("Need both target classes (0 and 1).")

    print(f"Labelled rows: {len(data):,}")
    print(f"Unique participants: {data[GROUP].nunique():,}")
    print("Target counts:")
    print(data[TARGET].value_counts().sort_index().to_string())

    print("\nCandidate input features:")
    for feature in FEATURES:
        print(f"  {feature}: {data[feature].notna().sum():,} non-missing")

    # Participant-level split: no participant appears in both sets.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(data[FEATURES], data[TARGET], groups=data[GROUP])
    )

    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    if set(train[GROUP]) & set(test[GROUP]):
        raise RuntimeError("Participant leakage detected in split.")
    if train[TARGET].nunique() != 2 or test[TARGET].nunique() != 2:
        raise ValueError(
            "Train/test split does not contain both classes. "
            "Adjust the split strategy before evaluating."
        )

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])

    # Install xgboost with:
    # python -m pip install xgboost
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed. Run: python -m pip install xgboost"
        ) from exc

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = negatives / max(positives, 1)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", min_samples_leaf=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            min_samples_leaf=3, n_jobs=-1, random_state=42
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced", max_iter=10000, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05,
            max_depth=2, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced",
            min_samples_leaf=3, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", n_jobs=-1,
            random_state=42
        ),
    }

    results = []
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        # Linear SVM has decision scores rather than probabilities.
        if hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(X_test)[:, 1]
        else:
            scores = pipeline.decision_function(X_test)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "Model": name,
            "Input_feature_count": len(FEATURES),
            "Train_rows": len(train),
            "Test_rows": len(test),
            "Train_participants": train[GROUP].nunique(),
            "Test_participants": test[GROUP].nunique(),
            "ROC_AUC": roc_auc_score(y_test, scores),
            "PR_AUC": average_precision_score(y_test, scores),
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Sensitivity": recall_score(
                y_test, predictions, zero_division=0
            ),
            "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "F1": f1_score(y_test, predictions, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
        print(
            f"{name}: ROC-AUC={results[-1]['ROC_AUC']:.3f}, "
            f"PR-AUC={results[-1]['PR_AUC']:.3f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        "PR_AUC", ascending=False
    )
    output = REPORTS / "treatment_proxy_7_model_comparison.csv"
    results_df.to_csv(output, index=False)

    print("\nComparison (sorted by PR-AUC):")
    print(results_df[
        ["Model", "ROC_AUC", "PR_AUC", "Sensitivity",
         "Specificity", "Precision", "F1"]
    ].to_string(index=False))
    print(f"\nSaved results to: {output}")
    print("\nFeatures used:", ", ".join(FEATURES))


if __name__ == "__main__":
    main()
