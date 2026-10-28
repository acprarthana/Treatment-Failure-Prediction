
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / (
    "hiv_longitudinal_outcomes_enriched_final_candidate.csv"
)
OUT = ROOT / "reports" / "treatment_proxy_threshold_results.csv"
TARGET = "treatment_outcome"
GROUP = "participant_id"

NUM = [
    "CD4", "viral_load", "year", "CD4_previous",
    "viral_load_previous", "CD4_change", "viral_load_change",
    "years_since_previous_visit",
]
CAT = ["sex", "VISIT"]
FEATURES = NUM + CAT


def main():
    df = pd.read_csv(DATA, low_memory=False)
    for c in ["CD4", "viral_load", "year", TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Recreate historical features in chronological order per participant.
    df["_year"] = df["year"]
    df["_visit"] = pd.to_numeric(df["VISIT"], errors="coerce")
    df = df.sort_values([GROUP, "_year", "_visit"], na_position="last").copy()
    g = df.groupby(GROUP, sort=False)
    df["CD4_previous"] = g["CD4"].shift(1)
    df["viral_load_previous"] = g["viral_load"].shift(1)
    prev_year = g["year"].shift(1)
    df["CD4_change"] = df["CD4"] - df["CD4_previous"]
    df["viral_load_change"] = df["viral_load"] - df["viral_load_previous"]
    df["years_since_previous_visit"] = df["year"] - prev_year

    data = df[df[TARGET].isin([0, 1]) & df[GROUP].notna()].copy()
    data[TARGET] = data[TARGET].astype(int)

    # Hold out test participants. Use only the remaining training participants
    # for threshold selection.
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(
        split.split(data[FEATURES], data[TARGET], groups=data[GROUP])
    )
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=123)
    fit_idx, val_idx = next(
        inner.split(train[FEATURES], train[TARGET], groups=train[GROUP])
    )
    fit = train.iloc[fit_idx]
    val = train.iloc[val_idx]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("num", num_pipe, NUM),
        ("cat", cat_pipe, CAT),
    ])

    neg = int((fit[TARGET] == 0).sum())
    pos = int((fit[TARGET] == 1).sum())

    models = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            min_samples_leaf=3, n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg / max(pos, 1),
            eval_metric="logloss", n_jobs=-1, random_state=42
        ),
    }

    rows = []
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline([("prep", clone(prep)), ("model", model)])
        pipe.fit(fit[FEATURES], fit[TARGET])
        val_scores = pipe.predict_proba(val[FEATURES])[:, 1]
        fitted[name] = pipe

        # Select threshold with best F1 on validation only.
        thresholds = np.arange(0.05, 0.96, 0.01)
        candidates = []
        for threshold in thresholds:
            pred = (val_scores >= threshold).astype(int)
            candidates.append((
                f1_score(val[TARGET], pred, zero_division=0),
                threshold
            ))
        best_f1, best_threshold = max(candidates, key=lambda x: x[0])

        # Refit on all training participants, then evaluate held-out test.
        final_pipe = Pipeline([("prep", clone(prep)), ("model", model)])
        final_pipe.fit(train[FEATURES], train[TARGET])
        test_scores = final_pipe.predict_proba(test[FEATURES])[:, 1]
        test_pred = (test_scores >= best_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            test[TARGET], test_pred, labels=[0, 1]
        ).ravel()

        rows.append({
            "Model": name,
            "Threshold_selected_on_validation": best_threshold,
            "Validation_F1_at_selected_threshold": best_f1,
            "Test_ROC_AUC": roc_auc_score(test[TARGET], test_scores),
            "Test_PR_AUC": average_precision_score(test[TARGET], test_scores),
            "Test_Precision": precision_score(
                test[TARGET], test_pred, zero_division=0
            ),
            "Test_Sensitivity": recall_score(
                test[TARGET], test_pred, zero_division=0
            ),
            "Test_Specificity": tn / (tn + fp) if tn + fp else np.nan,
            "Test_F1": f1_score(test[TARGET], test_pred, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })

    result = pd.DataFrame(rows).sort_values("Test_PR_AUC", ascending=False)
    OUT.parent.mkdir(exist_ok=True)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
