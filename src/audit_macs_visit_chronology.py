from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "interim" / "lab_rslt_clean.csv"
SUMMARY_FILE = ROOT / "data" / "interim" / "macs_chronology_audit_v2.csv"
WARNING_FILE = ROOT / "data" / "interim" / "macs_chronology_warnings_v2.csv"
SPECIAL_FILE = ROOT / "data" / "interim" / "macs_special_recruitment_audit_v2.csv"

SPECIAL_CODES = {"075", "365", "525"}

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)

required = {"CASEID", "VISIT", "FORM", "LDATY"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["CASEID"] = df["CASEID"].str.strip()
df["VISIT"] = df["VISIT"].str.strip().str.zfill(3)
df["FORM"] = df["FORM"].str.strip()
df["LDATY"] = pd.to_numeric(df["LDATY"].str.strip(), errors="coerce")

# One row per participant and visit for chronology checking.
# Multiple forms at the same visit/year are not separate visits.
visits = (
    df[["CASEID", "VISIT", "LDATY"]]
    .drop_duplicates()
    .dropna(subset=["LDATY"])
    .copy()
)

visits["LDATY"] = visits["LDATY"].astype(int)

special = visits[visits["VISIT"].isin(SPECIAL_CODES)].copy()
ordinary = visits[~visits["VISIT"].isin(SPECIAL_CODES)].copy()

# Report special recruitment-code observations separately.
special_summary = (
    special.groupby(["VISIT", "LDATY"])
    .size()
    .rename("participant_visit_year_rows")
    .reset_index()
)
special_summary.to_csv(SPECIAL_FILE, index=False)

# Chronology checks use ordinary visits only, ordered by numeric visit code.
ordinary["VISIT_NUM"] = pd.to_numeric(ordinary["VISIT"], errors="coerce")
ordinary = ordinary.dropna(subset=["VISIT_NUM"])
ordinary = ordinary.sort_values(["CASEID", "VISIT_NUM"])

ordinary["PREVIOUS_VISIT"] = ordinary.groupby("CASEID")["VISIT"].shift(1)
ordinary["PREVIOUS_YEAR"] = ordinary.groupby("CASEID")["LDATY"].shift(1)
ordinary["YEAR_CHANGE"] = ordinary["LDATY"] - ordinary["PREVIOUS_YEAR"]

warnings = ordinary[
    ordinary["PREVIOUS_YEAR"].notna()
    & (ordinary["YEAR_CHANGE"] < 0)
].copy()

warnings = warnings.rename(columns={"LDATY": "CURRENT_YEAR"})
warnings.to_csv(WARNING_FILE, index=False)

summary = [
    ("input_rows", len(df)),
    ("unique_participants", df["CASEID"].nunique()),
    ("special_recruitment_codes", len(SPECIAL_CODES)),
    ("special_code_participant_visit_year_rows", len(special)),
    ("ordinary_participant_visit_year_rows", len(ordinary)),
    ("ordinary_backwards_year_warnings", len(warnings)),
    ("original_warning_report_modified", "No"),
    ("raw_data_modified", "No"),
]

pd.DataFrame(summary, columns=["check", "value"]).to_csv(
    SUMMARY_FILE, index=False
)

print("Chronology audit v2 completed.")
print("\nSummary:")
print(pd.DataFrame(summary, columns=["check", "value"]).to_string(index=False))
print("\nOrdinary visit chronology warnings:")
print(warnings.to_string(index=False) if len(warnings) else "None")
print("\nSpecial recruitment code year summary:")
print(special_summary.to_string(index=False))
print("\nNew files saved:")
print(SUMMARY_FILE)
print(WARNING_FILE)
print(SPECIAL_FILE)
print("\nOriginal data and reports were not modified.")
