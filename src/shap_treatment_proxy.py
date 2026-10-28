from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATA = (
    ROOT
    / "data"
    / "interim"
    / "hiv_longitudinal_outcomes_enriched_final_candidate.csv"
)

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TARGET = "treatment_outcome"
GROUP = "participant_id"

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

CATEGORICAL_FEATURES = [
    "sex",
    "VISIT",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_dataset():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA, low_memory=False)

    required = [
        TARGET,
        GROUP,
        "CD4",
        "viral_load",
        "year",
        "sex",
        "VISIT",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    # Chronological ordering within participant.
    df["_year_order"] = pd.to_numeric(df["year"], errors="coerce")
    df["_visit_order"] = pd.to_numeric(df["VISIT"], errors="coerce")

    df = df.sort_values(
        [GROUP, "_year_order", "_visit_order"],
        na_position="last",
    ).copy()

    # Numeric conversion.
    for col in ["CD4", "viral_load", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Previous-visit features.
    grouped = df.groupby(GROUP, sort=False)

    df["CD4_previous"] = grouped["CD4"].shift(1)
    df["viral_load_previous"] = grouped["viral_load"].shift(1)
    df["_previous_year"] = grouped["year"].shift(1)

    df["CD4_change"] = (
        df["CD4"] - df["CD4_previous"]
    )

    df["viral_load_change"] = (
        df["viral_load"] - df["viral_load_previous"]
    )

    df["years_since_previous_visit"] = (
        df["year"] - df["_previous_year"]
    )

    # Keep labelled participant-level observations.
    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    data = df.loc[
        df[TARGET].isin([0, 1])
        & df[GROUP].notna()
    ].copy()

    data[TARGET] = data[TARGET].astype(int)

    if data[TARGET].nunique() != 2:
        raise ValueError(
            "Need both target classes (0 and 1)."
        )

    return data


def main():
    data = build_dataset()

    print(f"Labelled rows: {len(data):,}")
    print(
        f"Unique participants: "
        f"{data[GROUP].nunique():,}"
    )

    # Same participant-level split used by the existing
    # treatment proxy model evaluation.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            data[FEATURES],
            data[TARGET],
            groups=data[GROUP],
        )
    )

    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    if set(train[GROUP]) & set(test[GROUP]):
        raise RuntimeError(
            "Participant leakage detected."
        )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # Preprocessing matching the existing model pipeline.
    numeric_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocess = ColumnTransformer([
        (
            "numeric",
            numeric_pipe,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipe,
            CATEGORICAL_FEATURES,
        ),
    ])

    X_train_processed = preprocess.fit_transform(
        X_train
    )

    X_test_processed = preprocess.transform(
        X_test
    )

    # XGBoost configuration matching the existing
    # treatment proxy model.
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())

    scale_pos_weight = (
        negatives / max(positives, 1)
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train_processed,
        y_train,
    )

    # Recover the transformed feature names.
    feature_names = (
        preprocess
        .get_feature_names_out()
        .tolist()
    )

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    print(
        f"Transformed features: "
        f"{len(feature_names):,}"
    )

    # Native TreeExplainer for XGBoost.
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_test_processed
    )

    # Global mean absolute SHAP importance.
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": np.abs(
            shap_values
        ).mean(axis=0),
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )

    importance_path = (
        REPORTS
        / "shap_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    # Per-observation SHAP values.
    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names,
        index=X_test.index,
    )

    shap_path = (
        REPORTS
        / "shap_test_values.csv"
    )

    shap_df.to_csv(
        shap_path,
        index=False,
    )

    # Save test-set predictions alongside participant IDs.
    predictions = model.predict_proba(
        X_test_processed
    )[:, 1]

    prediction_output = test[
        [
            GROUP,
            "VISIT",
            "year",
            TARGET,
        ]
    ].copy()

    prediction_output[
        "predicted_probability"
    ] = predictions

    prediction_output_path = (
        REPORTS
        / "shap_test_predictions.csv"
    )

    prediction_output.to_csv(
        prediction_output_path,
        index=False,
    )

    # Human-readable report.
    report_path = (
        REPORTS
        / "shap_treatment_proxy_report.md"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# SHAP Explainability — "
            "Treatment Outcome Proxy\n\n"
        )

        f.write(
            "## Dataset and model\n\n"
        )

        f.write(
            f"- Labelled rows: "
            f"{len(data):,}\n"
        )

        f.write(
            f"- Training rows: "
            f"{len(train):,}\n"
        )

        f.write(
            f"- Test rows: "
            f"{len(test):,}\n"
        )

        f.write(
            f"- Training participants: "
            f"{train[GROUP].nunique():,}\n"
        )

        f.write(
            f"- Test participants: "
            f"{test[GROUP].nunique():,}\n"
        )

        f.write(
            "- Model: XGBoost\n"
        )

        f.write(
            "- Explainer: SHAP TreeExplainer\n"
        )

        f.write(
            "- Split: participant-level "
            "80/20 holdout\n\n"
        )

        f.write(
            "## Input variables\n\n"
        )

        for feature in FEATURES:
            f.write(
                f"- `{feature}`\n"
            )

        f.write(
            "\n## Top SHAP features\n\n"
        )

        f.write(
            "| Rank | Feature | "
            "Mean absolute SHAP |\n"
        )

        f.write(
            "|---:|---|---:|\n"
        )

        for rank, row in enumerate(
            importance.head(10).itertuples(
                index=False
            ),
            start=1,
        ):
            f.write(
                f"| {rank} | "
                f"{row.Feature} | "
                f"{row.Mean_Absolute_SHAP:.6f} |\n"
            )

        f.write(
            "\n## Important interpretation note\n\n"
        )

        f.write(
            "SHAP values explain the XGBoost model's "
            "predictions. They do not establish causal "
            "relationships or clinical treatment failure. "
            "The underlying target is a treatment-outcome "
            "proxy as documented by the existing evaluation "
            "pipeline.\n"
        )

    print(
        "\nSaved SHAP outputs:"
    )

    print(
        f"  {importance_path}"
    )

    print(
        f"  {shap_path}"
    )

    print(
        f"  {prediction_output_path}"
    )

    print(
        f"  {report_path}"
    )

    print(
        "\nTop SHAP features:"
    )

    print(
        importance.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()