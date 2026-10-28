
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
from sklearn.ensemble import GradientBoostingClassifier

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

# Same predictors used in the treatment-proxy model evaluation.
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


def build_dataset(df):
    """
    Reproduce the feature engineering used by the treatment-proxy model.
    """

    # Chronological ordering within participant.
    df["_year_order"] = pd.to_numeric(
        df["year"], errors="coerce"
    )
    df["_visit_order"] = pd.to_numeric(
        df["VISIT"], errors="coerce"
    )

    df = df.sort_values(
        [GROUP, "_year_order", "_visit_order"],
        na_position="last",
    ).copy()

    # Numeric measurements.
    for col in ["CD4", "viral_load", "year"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

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

    # Target.
    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="coerce"
    )

    # Keep labelled rows only.
    data = df.loc[
        df[TARGET].isin([0, 1])
        & df[GROUP].notna()
    ].copy()

    data[TARGET] = data[TARGET].astype(int)

    return data


def main():

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    print(f"Reading dataset: {DATA}")

    df = pd.read_csv(
        DATA,
        low_memory=False,
    )

    required = [
        TARGET,
        GROUP,
        "CD4",
        "viral_load",
        "year",
        "sex",
        "VISIT",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}"
        )

    data = build_dataset(df)

    print(
        f"Labelled rows: {len(data):,}"
    )

    print(
        f"Unique participants: "
        f"{data[GROUP].nunique():,}"
    )

    print("\nTarget counts:")
    print(
        data[TARGET]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------
    # Participant-level train/test split
    # ---------------------------------------------------------

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

    print(
        f"\nTrain rows: {len(train):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    print(
        f"Train participants: "
        f"{train[GROUP].nunique():,}"
    )

    print(
        f"Test participants: "
        f"{test[GROUP].nunique():,}"
    )

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

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
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
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

    X_train_raw = train[FEATURES]
    X_test_raw = test[FEATURES]

    y_train = train[TARGET]
    y_test = test[TARGET]

    # Fit preprocessing only on training data.
    X_train = preprocess.fit_transform(
        X_train_raw
    )

    X_test = preprocess.transform(
        X_test_raw
    )

    # ---------------------------------------------------------
    # Get transformed feature names
    # ---------------------------------------------------------

    feature_names = (
        preprocess
        .get_feature_names_out()
    )

    print(
        f"\nTransformed features: "
        f"{len(feature_names)}"
    )

    # SHAP works most naturally with dense arrays
    # for this relatively small test set.
    if hasattr(X_train, "toarray"):
        X_train_shap = X_train.toarray()
    else:
        X_train_shap = np.asarray(X_train)

    if hasattr(X_test, "toarray"):
        X_test_shap = X_test.toarray()
    else:
        X_test_shap = np.asarray(X_test)

    # ---------------------------------------------------------
    # Gradient Boosting model
    # ---------------------------------------------------------

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )

    print(
        "\nTraining Gradient Boosting model..."
    )

    model.fit(
        X_train_shap,
        y_train,
    )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------

    probabilities = model.predict_proba(
        X_test_shap
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ---------------------------------------------------------
    # SHAP explanation
    # ---------------------------------------------------------

    print(
        "\nCalculating SHAP values..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_test_shap
    )

    # Depending on SHAP/model versions,
    # binary classification may return either
    # an array or a list.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap,
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    importance_output = (
        REPORTS
        / "shap_feature_importance.csv"
    )

    importance.to_csv(
        importance_output,
        index=False,
    )

    # ---------------------------------------------------------
    # SHAP values for every test record
    # ---------------------------------------------------------

    shap_values_df = pd.DataFrame(
        shap_values,
        columns=feature_names,
    )

    shap_values_df.insert(
        0,
        "row_index",
        test.index.to_numpy(),
    )

    shap_values_output = (
        REPORTS
        / "shap_test_values.csv"
    )

    shap_values_df.to_csv(
        shap_values_output,
        index=False,
    )

    # ---------------------------------------------------------
    # Test predictions
    # ---------------------------------------------------------

    predictions_df = pd.DataFrame({
        "row_index": test.index.to_numpy(),
        "participant_id": test[GROUP].to_numpy(),
        "actual_treatment_outcome": y_test.to_numpy(),
        "predicted_probability": probabilities,
        "predicted_class": predictions,
    })

    predictions_output = (
        REPORTS
        / "shap_test_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_output,
        index=False,
    )

    # ---------------------------------------------------------
    # Markdown report
    # ---------------------------------------------------------

    report_output = (
        REPORTS
        / "shap_treatment_proxy_report.md"
    )

    top_features = importance.head(10)

    with open(
        report_output,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "# Gradient Boosting SHAP Explainability Report\n\n"
        )

        f.write(
            "## Model\n\n"
            "This report explains the Gradient Boosting "
            "treatment-outcome proxy model. Gradient Boosting "
            "was selected as the best-performing model among "
            "the evaluated models based on test PR-AUC.\n\n"
        )

        f.write(
            "## Dataset\n\n"
        )

        f.write(
            f"- Labelled rows: {len(data):,}\n"
            f"- Unique participants: "
            f"{data[GROUP].nunique():,}\n"
            f"- Training rows: {len(train):,}\n"
            f"- Test rows: {len(test):,}\n"
            f"- Transformed features: "
            f"{len(feature_names)}\n"
            "- Split: participant-level holdout\n"
            "- Random state: 42\n\n"
        )

        f.write(
            "## Original input features\n\n"
        )

        for feature in FEATURES:
            f.write(
                f"- `{feature}`\n"
            )

        f.write("\n")

        f.write(
            "## Top SHAP features\n\n"
        )

        f.write(
            "| Rank | Feature | "
            "Mean Absolute SHAP |\n"
        )

        f.write(
            "|---:|---|---:|\n"
        )

        for rank, row in (
            top_features
            .reset_index(drop=True)
            .iterrows()
        ):
            f.write(
                f"| {rank + 1} | "
                f"{row['Feature']} | "
                f"{row['Mean_Absolute_SHAP']:.6f} |\n"
            )

        f.write("\n")

        f.write(
            "## Interpretation\n\n"
        )

        f.write(
            "Mean absolute SHAP values quantify how strongly "
            "each transformed feature contributed to the model's "
            "predictions on the test set. Larger values indicate "
            "greater average contribution to the model output; "
            "they do not by themselves establish clinical "
            "causality or treatment effectiveness.\n\n"
        )

        f.write(
            "## Important limitation\n\n"
            "The target is a treatment-outcome proxy based on the "
            "available labelled data and is not independently "
            "confirmed clinical treatment failure. SHAP describes "
            "the behaviour of the trained predictive model and "
            "should not be interpreted as evidence of causal "
            "relationships.\n"
        )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print("\nSaved SHAP outputs:")

    print(
        f"  {importance_output}"
    )

    print(
        f"  {shap_values_output}"
    )

    print(
        f"  {predictions_output}"
    )

    print(
        f"  {report_output}"
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
