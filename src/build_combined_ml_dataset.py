
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim"


def clean_code(series):
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def build_cohort(cohort):
    if cohort == "MACS":
        labs = pd.read_csv(DATA / "lab_rslt_clean.csv", dtype={"CASEID": str, "VISIT": str})
        drugs = pd.read_csv(DATA / "macs_drugf1_selected_v2.csv", dtype={"CASEID": str, "VISIT": str})
        cd4_col, year_col = "TCELL", "LDATY"
        drugs["failure_signal"] = pd.concat(
            [clean_code(drugs[c]) for c in ["STOT1", "STOT2", "STOT3"]], axis=1
        ).eq("65").any(axis=1)
    else:
        labs = pd.read_csv(DATA / "wihs_labsum_clean.csv", dtype={"CASEID": str, "VISIT": str})
        drugs = pd.read_csv(DATA / "wihs_drug1_clean.csv", dtype={"CASEID": str, "VISIT": str})
        cd4_col, year_col = "CD4N", None
        # WIHS medication stop-reason semantics have not been verified.
        drugs["failure_signal"] = False

    # Summarize labs to one row per participant-visit.
    labs["CD4"] = pd.to_numeric(labs[cd4_col], errors="coerce")
    labs["viral_load"] = pd.to_numeric(labs["VLOAD"], errors="coerce")
    labs["year"] = pd.to_numeric(labs[year_col], errors="coerce") if year_col else pd.NA

    lab_visit = (
        labs.groupby(["CASEID", "VISIT"], as_index=False)
        .agg(
            CD4=("CD4", "median"),
            viral_load=("viral_load", "median"),
            year=("year", "median"),
            lab_records=("CASEID", "size"),
        )
    )

    # Summarize medication signals separately before joining.
    drug_visit = (
        drugs.groupby(["CASEID", "VISIT"], as_index=False)
        .agg(
            medication_records=("CASEID", "size"),
            reported_failure_signal=("failure_signal", "max"),
        )
    )

    result = lab_visit.merge(
        drug_visit,
        on=["CASEID", "VISIT"],
        how="left",
        validate="one_to_one",
    )

    result["cohort"] = cohort
    result["sex"] = "Male" if cohort == "MACS" else "Female"
    result["participant_id"] = cohort + "_" + result["CASEID"]
    result["outcome"] = pd.NA
    result["outcome_status"] = "unknown"

    result.loc[
        result["reported_failure_signal"].eq(True),
        "outcome_status",
    ] = "reported_failure_related_signal"

    return result


macs = build_cohort("MACS")
wihs = build_cohort("WIHS")
combined = pd.concat([macs, wihs], ignore_index=True)

# Final structural safeguards.
assert not combined.duplicated(["participant_id", "VISIT"]).any()
assert combined["participant_id"].notna().all()

out = DATA / "combined_ml_dataset_provisional.csv"
combined.to_csv(out, index=False)

print("Saved:", out)
print("Rows:", len(combined))
print("Unique participants:", combined["participant_id"].nunique())
print("\nCohort summary:")
print(combined.groupby("cohort").agg(
    rows=("participant_id", "size"),
    participants=("participant_id", "nunique"),
    CD4_available=("CD4", "count"),
    viral_load_available=("viral_load", "count"),
    medication_matched=("medication_records", "count"),
).to_string())
print("\nOutcome labels remain provisional/unknown; do not train yet.")
