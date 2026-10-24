from pathlib import Path
import pandas as pd

raw_path = Path("data/raw/WIHS/WIHS PDS/CSV/labsum.csv")
out_dir = Path("data/interim")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw_path, encoding="cp1252", low_memory=False)

# Preserve original row count and participant-visit structure.
original_rows = len(df)
original_columns = list(df.columns)

# The source stores these laboratory dates as years.
for col in ["VLDT", "CD4DT"]:
    years = pd.to_numeric(df[col], errors="coerce")
    df[col] = years.astype("Int64")

# Add a review flag; do not delete or impute these records.
df["REVIEW_MISSING_UD_VL"] = (
    df["VLOAD"].notna() & df["UD_VL"].isna()
)

# Save a separate interim dataset.
clean_path = out_dir / "wihs_labsum_clean.csv"
df.to_csv(clean_path, index=False)

# Build a transparent validation report.
report = []

def add_check(check, observed, expected, status, note):
    report.append({
        "check": check,
        "observed": observed,
        "expected": expected,
        "status": status,
        "note": note
    })

add_check(
    "row_count_preserved", len(df), original_rows,
    "PASS" if len(df) == original_rows else "FAIL",
    "No rows intentionally removed"
)
add_check(
    "original_columns_preserved",
    len([c for c in original_columns if c in df.columns]),
    len(original_columns),
    "PASS" if all(c in df.columns for c in original_columns) else "FAIL",
    "Original columns retained; one review flag added"
)
add_check(
    "duplicate_caseid_visit",
    int(df.duplicated(["CASEID", "VISIT"]).sum()),
    0,
    "PASS" if not df.duplicated(["CASEID", "VISIT"]).any() else "REVIEW",
    "Review duplicate keys if present"
)
add_check(
    "viral_load_present_status_missing",
    int(df["REVIEW_MISSING_UD_VL"].sum()),
    0,
    "REVIEW" if df["REVIEW_MISSING_UD_VL"].any() else "PASS",
    "Retained records; no imputation"
)

for col in ["VLDT", "CD4DT"]:
    years = df[col]
    invalid = years.notna() & ((years < 1994) | (years > 2015))
    add_check(
        f"{col}_valid_year_range",
        int(invalid.sum()),
        0,
        "PASS" if not invalid.any() else "FAIL",
        "Documented year range: 1994–2015"
    )

report_path = out_dir / "wihs_labsum_validation_report.csv"
pd.DataFrame(report).to_csv(report_path, index=False)

print("Saved cleaned interim file:", clean_path)
print("Saved validation report:", report_path)
print("Rows:", len(df))
print("Review flags:", int(df["REVIEW_MISSING_UD_VL"].sum()))
print("\nValidation report:")
print(pd.DataFrame(report).to_string(index=False))
