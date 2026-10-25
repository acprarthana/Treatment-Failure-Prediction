from pathlib import Path
import pandas as pd

raw_path = Path("data/raw/WIHS/WIHS PDS/CSV/drug1.csv")
out_dir = Path("data/interim")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw_path, encoding="cp1252", low_memory=False)
original_rows = len(df)
original_columns = list(df.columns)

# Convert date-like numeric fields to nullable integers without
# changing their meaning or replacing documented special codes.
for col in ["INITD1", "DCMPD1", "BGNMD1", "BGNYD1",
            "LASMD1", "LASYD1"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

# Review flags: preserve suspicious values for manual inspection.
df["REVIEW_DCMPD1_YEAR"] = (
    df["DCMPD1"].notna()
    & ~df["DCMPD1"].between(1993, 2015)
    & ~df["DCMPD1"].isin([-1, -7, -8, -9])
)

df["REVIEW_BGNYD1_YEAR"] = (
    df["BGNYD1"].notna()
    & ~df["BGNYD1"].between(1994, 2029)
    & ~df["BGNYD1"].isin([-1, -7, -8, -9])
)

# Document fully missing fields; do not invent values.
df["REVIEW_INITD1_ALL_MISSING"] = df["INITD1"].isna().all()
df["REVIEW_BGNMD1_ALL_MISSING"] = df["BGNMD1"].isna().all()

# Exact duplicates are flagged, never automatically removed.
df["REVIEW_EXACT_DUPLICATE"] = df.duplicated(keep=False)

# Multiple records for a participant and visit are expected in a
# medication-level file and are not treated as duplicates.
df["REVIEW_CASEID_VISIT_REPEAT"] = df.duplicated(
    ["CASEID", "VISIT"], keep=False
)

clean_path = out_dir / "wihs_drug1_clean.csv"
df.to_csv(clean_path, index=False)

checks = [
    ("row_count_preserved", len(df), original_rows),
    ("original_columns_preserved",
     sum(c in df.columns for c in original_columns), len(original_columns)),
    ("exact_duplicate_rows", int(df.duplicated(
        subset=original_columns).sum()), 0),
    ("missing_CASEID", int(df["CASEID"].isna().sum()), 0),
    ("missing_VISIT", int(df["VISIT"].isna().sum()), 0),
    ("invalid_VTYP", int((~df["VTYP"].isin([1])).sum()), 0),
    ("DCMPD1_outside_documented_range",
     int(df["REVIEW_DCMPD1_YEAR"].sum()), 0),
    ("BGNYD1_outside_documented_range",
     int(df["REVIEW_BGNYD1_YEAR"].sum()), 0),
]

report = pd.DataFrame(checks, columns=["check", "observed", "expected"])
report["status"] = report.apply(
    lambda r: "PASS" if r["observed"] == r["expected"] else "REVIEW",
    axis=1
)
report["note"] = (
    "Review flagged values against the codebook; "
    "no rows deleted and no values imputed"
)

report_path = out_dir / "wihs_drug1_validation_report.csv"
report.to_csv(report_path, index=False)

print("Saved:", clean_path)
print("Saved:", report_path)
print("\nROWS:", len(df))
print("UNIQUE CASEID:", df["CASEID"].nunique())
print("EXACT DUPLICATE ROWS:",
      int(df.duplicated(subset=original_columns).sum()))
print("\nREVIEW FLAGS:")
flag_cols = [c for c in df.columns if c.startswith("REVIEW_")]
print(df[flag_cols].sum().to_string())
print("\nVALIDATION REPORT:")
print(report.to_string(index=False))
