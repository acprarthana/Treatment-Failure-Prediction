
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim"

SOURCE = DATA / "hiv_longitudinal_outcomes_enriched_datefix.csv"
OUTPUT = DATA / "hiv_future_virologic_prediction.csv"
REPORT = DATA / "hiv_prediction_dataset_summary.txt"

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing repaired dataset: {SOURCE}")

df = pd.read_csv(SOURCE, low_memory=False)

required = {"cohort", "participant_id", "year", "viral_load", "CD4"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

for col in ["year", "viral_load", "CD4"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Use only measured viral loads with a valid participant and year.
# The target is the next observed visit's virologic status, not
# a clinically confirmed ART treatment-failure diagnosis.
df = df.dropna(subset=["cohort", "participant_id", "year"])
df = df.sort_values(["cohort", "participant_id", "year", "VISIT"])

# If multiple records exist for one participant in the same year,
# retain the last available row for a year-level prediction dataset.
df = df.drop_duplicates(
    subset=["cohort", "participant_id", "year"], keep="last"
)

# Keep only rows with numeric viral load for target construction.
df = df.dropna(subset=["viral_load"]).copy()

# This is an exploratory threshold, not a validated clinical endpoint.
# Values at assay detection limits may be censored or specially coded.
# Therefore, exclude common MACS special codes from target assignment.
df["target_measurement_usable"] = True
is_macs = df["cohort"].eq("MACS")

# In standardized MACS VLOAD, 10 means <20 and 40 means <50.
# Both are safely below the exploratory 200 copies/mL threshold.
# Keep 1, 300, and 400 unclassified until their meanings in the
# standardized VLOAD field are verified from the complete codebook.
df.loc[
    is_macs & df["viral_load"].isin([1, 300, 400]),
    "target_measurement_usable"
] = False

# For WIHS, use detection status and assay limits where available.
# Match original lab metadata by participant ID and visit key.
labs_path = DATA / "wihs_labsum_clean.csv"
if labs_path.exists():
    labs = pd.read_csv(labs_path, low_memory=False)
    for col in ["CASEID", "VISIT"]:
        if col in labs.columns:
            labs[col] = (
                labs[col].astype("string").str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )
        if col in df.columns:
            df[col] = (
                df[col].astype("string").str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )

    meta_cols = [
        c for c in ["CASEID", "VISIT", "UD_VL", "LLQ_VL"]
        if c in labs.columns
    ]
    if {"CASEID", "VISIT"}.issubset(meta_cols):
        meta = labs[meta_cols].drop_duplicates(["CASEID", "VISIT"])
        df = df.merge(
            meta, on=["CASEID", "VISIT"], how="left",
            validate="many_to_one"
        )

        wihs = df["cohort"].eq("WIHS")
        if "UD_VL" in df.columns:
            # UD_VL=1 means below quantification; do not treat the
            # recorded limit value as an exact measured viral load.
            df.loc[wihs & df["UD_VL"].eq(1), "target_measurement_usable"] = False
            # UD_VL=3 is above the assay's upper quantification limit;
            # retain as high only if the value is recorded numerically.
        if "LLQ_VL" in df.columns:
            df["LLQ_VL"] = pd.to_numeric(df["LLQ_VL"], errors="coerce")

# A high/low research target requires an explicit numeric result.
# We use 200 copies/mL as an exploratory threshold only where values
# are not known special codes or below-quantification observations.
df["virologic_status"] = np.nan
usable = df["target_measurement_usable"]
df.loc[usable & (df["viral_load"] < 200), "virologic_status"] = 0
df.loc[usable & (df["viral_load"] >= 200), "virologic_status"] = 1

# Predict the next observed year-level result using current/past features.
# Require a later observation in a different year.
df = df.sort_values(["cohort", "participant_id", "year"])
group = df.groupby(["cohort", "participant_id"], sort=False)

df["future_year"] = group["year"].shift(-1)
df["future_virologic_status"] = group["virologic_status"].shift(-1)
df["future_viral_load"] = group["viral_load"].shift(-1)

# Keep only pairs where the next observed measurement is 1–2 years later.
gap = df["future_year"] - df["year"]
df["year_gap_to_target"] = gap
df = df[gap.between(1, 2)].copy()

# Predictors are strictly current-visit variables; future columns are labels/audit.
feature_cols = [
    c for c in [
        "cohort", "participant_id", "CASEID", "VISIT", "year",
        "CD4", "viral_load", "sex", "lab_records", "medication_records"
    ] if c in df.columns
]
target_cols = [
    "future_year", "future_viral_load",
    "future_virologic_status", "year_gap_to_target"
]

result = df[feature_cols + target_cols].copy()
result = result.rename(columns={
    "year": "index_year",
    "CD4": "index_CD4",
    "viral_load": "index_viral_load",
    "future_year": "target_year",
    "future_viral_load": "target_viral_load",
    "future_virologic_status": "target_future_virologic_status"
})

# Keep labeled rows only in the modeling file.
result = result.dropna(subset=["target_future_virologic_status"])
result["target_future_virologic_status"] = (
    result["target_future_virologic_status"].astype(int)
)

result.to_csv(OUTPUT, index=False)

lines = [
    "HIV FUTURE VIROLOGIC PREDICTION DATASET",
    f"Source rows: {len(pd.read_csv(SOURCE, usecols=['cohort']))}",
    f"Prediction rows: {len(result)}",
    f"Unique participants: {result['participant_id'].nunique()}",
    "",
    "Class counts:",
    result["target_future_virologic_status"].value_counts().sort_index().to_string(),
    "",
    "Rows by cohort:",
    result.groupby("cohort").size().to_string(),
    "",
    "IMPORTANT LIMITATIONS:",
    "- This is an exploratory future virologic-status target, not confirmed ART treatment failure.",
    "- Threshold 200 copies/mL is a research choice and requires clinical/source validation.",
    "- MACS special-coded viral-load values and WIHS below-quantification values are excluded from target assignment.",
    "- The next observed measurement is used; the 1–2 year gap is approximate because source dates are year-level.",
    "- Split train/test by participant, never by row, to prevent participant leakage.",
    "- Do not use target_year, target_viral_load, or target_future_virologic_status as predictors.",
]
REPORT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print(f"\nSaved: {OUTPUT}")
print(f"Saved: {REPORT}")
