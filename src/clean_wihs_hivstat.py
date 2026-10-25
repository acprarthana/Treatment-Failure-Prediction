from pathlib import Path
import pandas as pd

raw_path = Path("data/raw/WIHS/WIHS PDS/CSV/hivstat.csv")
out_dir = Path("data/interim")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw_path, encoding="cp1252", low_memory=False)
original_rows = len(df)
original_columns = list(df.columns)

# Convert year-like date fields and visit fields to nullable integers.
for col in ["LNEGDATE", "FPOSDATE", "NEGVIS", "POSVIS"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

# Flag values outside documented codebook ranges.
df["REVIEW_STATUS_CODE"] = ~df["STATUS"].isin([1, 2, 4, 5])

df["REVIEW_LNEGDATE_YEAR"] = (
    df["LNEGDATE"].notna()
    & ~df["LNEGDATE"].between(1994, 2015)
)

# Flag dates outside the codebook's documented range.
# Historical values are preserved; this flag does not prove they are errors.
df["REVIEW_FPOSDATE_YEAR"] = (
    df["FPOSDATE"].notna()
    & ~df["FPOSDATE"].between(1994, 2015)
)

df["REVIEW_NEGVIS_RANGE"] = (
    df["NEGVIS"].notna()
    & ~df["NEGVIS"].between(1, 42)
)

# POSVIS allows visit numbers 1–42 and code 88 for STATUS=5.
df["REVIEW_POSVIS_CODE"] = (
    df["POSVIS"].notna()
    & ~df["POSVIS"].between(1, 42)
    & ~((df["POSVIS"] == 88) & (df["STATUS"] == 5))
)

# Expected presence patterns based on the documented status meanings.
df["REVIEW_STATUS_FIELDS"] = False

negative = df["STATUS"] == 1
prevalent = df["STATUS"] == 2
converter = df["STATUS"].isin([4, 5])

# Negative: last-negative information expected, first-positive absent.
df.loc[negative, "REVIEW_STATUS_FIELDS"] = (
    df.loc[negative, ["LNEGDATE", "NEGVIS"]].isna().any(axis=1)
    | df.loc[negative, ["FPOSDATE", "POSVIS"]].notna().any(axis=1)
)

# Prevalent: first-positive information expected, last-negative absent.
df.loc[prevalent, "REVIEW_STATUS_FIELDS"] = (
    df.loc[prevalent, ["FPOSDATE", "POSVIS"]].isna().any(axis=1)
    | df.loc[prevalent, ["LNEGDATE", "NEGVIS"]].notna().any(axis=1)
)

# Converters: inspect all four fields manually; do not assume all must be present.
df.loc[converter, "REVIEW_STATUS_FIELDS"] = (
    df.loc[converter, ["LNEGDATE", "NEGVIS", "FPOSDATE", "POSVIS"]]
      .isna().any(axis=1)
)

clean_path = out_dir / "wihs_hivstat_clean.csv"
df.to_csv(clean_path, index=False)

# Report validation results.
checks = [
    ("row_count_preserved", len(df), original_rows),
    ("original_columns_preserved",
     sum(c in df.columns for c in original_columns), len(original_columns)),
    ("duplicate_CASEID", int(df["CASEID"].duplicated().sum()), 0),
    ("invalid_STATUS_codes", int(df["REVIEW_STATUS_CODE"].sum()), 0),
    ("invalid_LNEGDATE_years", int(df["REVIEW_LNEGDATE_YEAR"].sum()), 0),
    ("FPOSDATE_outside_documented_range", int(df["REVIEW_FPOSDATE_YEAR"].sum()), 0),
    ("invalid_NEGVIS_codes", int(df["REVIEW_NEGVIS_RANGE"].sum()), 0),
    ("invalid_POSVIS_codes", int(df["REVIEW_POSVIS_CODE"].sum()), 0),
    ("status_field_patterns_review",
     int(df["REVIEW_STATUS_FIELDS"].sum()), 0),
]

report = pd.DataFrame(checks, columns=["check", "observed", "expected"])
report["status"] = report.apply(
    lambda r: "PASS" if r["observed"] == r["expected"] else "REVIEW",
    axis=1
)
report["note"] = "Review flagged records; no automatic row deletion or imputation"

report_path = out_dir / "wihs_hivstat_validation_report.csv"
report.to_csv(report_path, index=False)

print("Saved:", clean_path)
print("Saved:", report_path)
print("\nSTATUS COUNTS:")
print(df["STATUS"].value_counts().sort_index().to_string())
print("\nFLAG COUNTS:")
flag_cols = [c for c in df.columns if c.startswith("REVIEW_")]
print(df[flag_cols].sum().to_string())
print("\nVALIDATION REPORT:")
print(report.to_string(index=False))

