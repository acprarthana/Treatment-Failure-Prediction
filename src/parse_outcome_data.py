
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

OUTCOME_FILE = ROOT / "data" / "raw" / "data" / "outcome.dat"
LAB_FILE = ROOT / "data" / "interim" / "lab_rslt_clean.csv"
OUTPUT_FILE = ROOT / "data" / "interim" / "outcome_selected.csv"

# Official SAS layout positions are 1-based and inclusive.
# These are selected fields for initial validation, not the full dataset.
FIELDS = {
    "CASEID": (1, 5),
    "REPDATEmm": (6, 7),
    "REPDATEyy": (8, 11),
    "AIDSCASE": (12, 12),
    "TAIDS": (13, 13),
    "DETERM": (14, 14),
    "AIDSREPDmm": (16, 17),
    "AIDSREPDyy": (18, 21),
    "DATE1mm": (29, 30),
    "DATE1yy": (31, 34),
    "DEATH": (937, 937),
    "DTHDATEmm": (938, 939),
    "DTHDATEyy": (940, 943),
    "DTHREPDTmm": (944, 945),
    "DTHREPDTyy": (946, 949),
}


def extract_field(record, start, end):
    """Extract a 1-based, inclusive field while preserving text and IDs."""
    return record[start - 1:end].strip()


def main():
    if not OUTCOME_FILE.exists():
        raise FileNotFoundError(OUTCOME_FILE)

    if not LAB_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned lab file not found: {LAB_FILE}"
        )

    # Keep raw records as text; never modify the original file.
    with OUTCOME_FILE.open(
        "r", encoding="ascii", errors="replace"
    ) as f:
        records = [line.rstrip("\r\n") for line in f]

    print("Raw outcome records:", len(records))
    print("Record lengths:", pd.Series(
        [len(r) for r in records]
    ).value_counts().to_dict())

    # Confirm the fixed-width layout before extracting fields.
    bad_lengths = [
        (i + 1, len(r))
        for i, r in enumerate(records)
        if len(r) != 1292
    ]

    if bad_lengths:
        raise ValueError(
            f"Unexpected outcome record lengths: {bad_lengths[:10]}"
        )

    data = {
        name: [
            extract_field(record, start, end)
            for record in records
        ]
        for name, (start, end) in FIELDS.items()
    }

    outcome = pd.DataFrame(data)

    # Empty source fields become missing values.
    outcome = outcome.replace("", pd.NA)

    # Verify the participant IDs are preserved exactly.
    if outcome["CASEID"].isna().any():
        raise ValueError("Missing CASEID values in outcome data")

    if not outcome["CASEID"].str.fullmatch(r"\d{5}").all():
        raise ValueError("Some outcome CASEIDs are not five digits")

    print("\nOutcome dataset shape:", outcome.shape)
    print("Unique outcome participants:", outcome["CASEID"].nunique())
    print("Duplicate outcome CASEIDs:", outcome["CASEID"].duplicated().sum())

    # This file is expected to be participant-level.
    # Stop rather than silently dropping duplicate participant records.
    if outcome["CASEID"].duplicated().any():
        print(
            "WARNING: duplicate CASEIDs exist. "
            "Investigate before joining."
        )

    print("\nAIDSCASE values:")
    print(outcome["AIDSCASE"].value_counts(dropna=False).sort_index())

    print("\nTAIDS values:")
    print(outcome["TAIDS"].value_counts(dropna=False).sort_index())

    print("\nDETERM values:")
    print(outcome["DETERM"].value_counts(dropna=False).sort_index())

    print("\nDEATH values:")
    print(outcome["DEATH"].value_counts(dropna=False).sort_index())

    # Compare participant sets; do not merge clinical labels yet.
    lab = pd.read_csv(
        LAB_FILE,
        dtype={"CASEID": str, "VISIT": str},
        keep_default_na=False,
        usecols=["CASEID", "VISIT"],
    )

    lab_ids = set(lab["CASEID"].unique())
    outcome_ids = set(outcome["CASEID"].unique())

    print("\nParticipant overlap")
    print("Unique lab participants:", len(lab_ids))
    print("Unique outcome participants:", len(outcome_ids))
    print("IDs in both files:", len(lab_ids & outcome_ids))
    print("Lab-only IDs:", len(lab_ids - outcome_ids))
    print("Outcome-only IDs:", len(outcome_ids - lab_ids))

    # Save only selected fields for inspection.
    outcome.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved selected outcome fields:", OUTPUT_FILE)
    print("Raw outcome file was not modified.")
    print("No treatment-failure label has been created.")


if __name__ == "__main__":
    main()
