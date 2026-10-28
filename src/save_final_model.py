from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "interim" / (
    "hiv_longitudinal_outcomes_enriched_final_candidate.csv"
)

MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS / "gradient_boosting_treatment_proxy.joblib"

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


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    print(f"Reading dataset: {DATA}")

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

    # ---------------------------------------------------------
    # Recreate chronological longitudinal features
    # ---------------------------------------------------------

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

    for col in ["CD4", "viral_load", "year"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    grouped = df.groupby(
        GROUP,
        sort=False,
    )

    df["CD4_previous"] = grouped["CD4"].shift(1)

    df["viral_load_previous"] = grouped[
        "viral_load"
    ].shift(1)

    df["_previous_year"] = grouped[
        "year"
    ].shift(1)

    df["CD4_change"] = (
        df["CD4"] -
        df["CD4_previous"]
    )

    df["viral_load_change"] = (
        df["viral_load"] -
        df["viral_load_previous"]
    )

    df["years_since_previous_visit"] = (
        df["year"] -
        df["_previous_year"]
    )

    # ---------------------------------------------------------
    # Keep valid treatment-outcome labels
    # ---------------------------------------------------------

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
            "Need both treatment outcome classes."
        )

    print(f"Labelled rows: {len(data):,}")
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
                handle_unknown="ignore"
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

    # ---------------------------------------------------------
    # Final Gradient Boosting model
    # ---------------------------------------------------------

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )

    pipeline = Pipeline([
        (
            "preprocess",
            preprocess,
        ),
        (
            "model",
            model,
        ),
    ])

    X = data[FEATURES]
    y = data[TARGET]

    print("\nTraining final Gradient Boosting model...")

    pipeline.fit(X, y)

    # ---------------------------------------------------------
    # Save complete pipeline
    # ---------------------------------------------------------

    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "model_name": "Gradient Boosting",
        "random_state": 42,
        "training_rows": len(data),
        "training_participants": data[GROUP].nunique(),
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print("\nModel saved successfully:")
    print(MODEL_PATH)

    print("\nFeatures:")
    for feature in FEATURES:
        print(f"  {feature}")


if __name__ == "__main__":
    main()