from pathlib import Path
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "hiv_longitudinal_outcomes_enriched_final_candidate.csv"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, low_memory=False)

# Only MACS rows with known proxy labels
df = df[
    df["cohort"].eq("MACS")
    & df["treatment_outcome"].isin([0, 1])
].copy()

# Do not use outcome-derived or stop-reason fields as predictors
numeric = ["CD4", "viral_load", "year"]
categorical = ["sex"]
features = numeric + categorical
target = "treatment_outcome"
group = "participant_id"

# Remove rows without participant ID
df = df.dropna(subset=[group])
df[target] = df[target].astype(int)

# Hold out whole participants
splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)
train_idx, test_idx = next(
    splitter.split(df, df[target], groups=df[group])
)

train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

assert set(train[group]).isdisjoint(set(test[group])), \
    "Participant leakage detected!"

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training participants:", train[group].nunique())
print("Testing participants:", test[group].nunique())
print("\nTrain label counts:\n", train[target].value_counts())
print("\nTest label counts:\n", test[target].value_counts())

preprocess = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
}

results = []
for name, estimator in models.items():
    model = Pipeline([
        ("preprocess", preprocess),
        ("model", estimator),
    ])
    model.fit(train[features], train[target])

    pred = model.predict(test[features])
    score = model.predict_proba(test[features])[:, 1]

    if test[target].nunique() < 2:
        raise ValueError("Test set has only one class; ROC-AUC is undefined.")

    tn, fp, fn, tp = confusion_matrix(
        test[target], pred, labels=[0, 1]
    ).ravel()

    result = {
        "model": name,
        "roc_auc": roc_auc_score(test[target], score),
        "pr_auc": average_precision_score(test[target], score),
        "sensitivity": tp / (tp + fn) if tp + fn else 0,
        "specificity": tn / (tn + fp) if tn + fp else 0,
        "test_rows": len(test),
        "test_participants": test[group].nunique(),
    }
    results.append(result)

    print(f"\n--- {name} ---")
    print(pd.Series(result).to_string())
    print("\nClassification report:")
    print(classification_report(
        test[target], pred, zero_division=0
    ))

results_df = pd.DataFrame(results)
path = OUT / "treatment_proxy_baseline_metrics.csv"
results_df.to_csv(path, index=False)
print("\nSaved:", path)
print(results_df.to_string(index=False))
print(
    "\nInterpretation: exploratory prediction of a MACS code-65 "
    "treatment-outcome proxy, not confirmed clinical treatment failure."
)
