"""
Finalize documented AIDS/death outcomes for the HIV longitudinal mini-project.

Inputs (existing files are not modified):
  data/interim/combined_ml_dataset_provisional.csv
  data/interim/outcome_selected.csv                 (MACS outcome extract)
  data/raw/WIHS/WIHS PDS/CSV/outcome.csv             (WIHS outcome records)

Outputs:
  data/interim/hiv_longitudinal_outcomes_enriched.csv
  reports/outcome_merge_validation.txt
  reports/outcome_merge_validation_by_cohort.csv

Important:
- This attaches source-documented outcome information; it does NOT claim ART treatment failure.
- Missing outcomes remain unknown, not negative.
- It does NOT assign a visit-level future-prediction label. That requires a justified prediction horizon
  and follow-up/censoring rules.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

BASE_FILE = DATA / "combined_ml_dataset_provisional.csv"
MACS_OUTCOME_FILE = DATA / "outcome_selected.csv"
WIHS_OUTCOME_FILE = RAW / "WIHS" / "WIHS PDS" / "CSV" / "outcome.csv"

def clean_id(s):
    # Preserve identifiers as strings; normalize whitespace and spreadsheet-style trailing .0.
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def numeric(s):
    return pd.to_numeric(s.replace({".": pd.NA, "": pd.NA, "-9": pd.NA, "-1": pd.NA}), errors="coerce")

def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")

def main():
    for p in [BASE_FILE, MACS_OUTCOME_FILE, WIHS_OUTCOME_FILE]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    REPORTS.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(BASE_FILE, dtype={"CASEID": "string", "VISIT": "string"}, low_memory=False)
    require_columns(base, ["CASEID", "VISIT", "cohort", "participant_id"], "Combined dataset")
    base["CASEID"] = clean_id(base["CASEID"])
    base["cohort"] = base["cohort"].astype("string").str.strip().str.upper()

    # MACS: one participant-level record per CASEID in the selected outcome extract.
    macs = pd.read_csv(MACS_OUTCOME_FILE, dtype={"CASEID": "string"}, low_memory=False)
    require_columns(
        macs,
        ["CASEID", "AIDSCASE", "TAIDS", "AIDSREPDyy", "DATE1yy", "DEATH", "DTHDATEyy"],
        "MACS outcome extract",
    )
    macs["CASEID"] = clean_id(macs["CASEID"])
    for c in ["AIDSCASE", "TAIDS", "AIDSREPDyy", "DATE1yy", "DEATH", "DTHDATEyy"]:
        macs[c] = numeric(macs[c])

    # Codebook: AIDSCASE=1 no CDC AIDS; 2 CDC AIDS diagnosis; 3 AIDS by death only.
    # Keep the source code and create a clearly named derived field.
    macs["macs_aids_documented"] = macs["AIDSCASE"].map({1: 0, 2: 1, 3: 1})
    macs["macs_aids_status"] = macs["AIDSCASE"].map({
        1: "no_CDC_AIDS_documented",
        2: "CDC_AIDS_diagnosis",
        3: "AIDS_by_death_only",
    }).fillna("unknown")
    # Prefer the first AIDS diagnosis year; use reported AIDS year as fallback.
    macs["macs_aids_year"] = macs["DATE1yy"].combine_first(macs["AIDSREPDyy"])
    macs["macs_death_type"] = macs["DEATH"]
    macs["macs_death_year"] = macs["DTHDATEyy"]
    macs["macs_outcome_record_present"] = True

    macs_cols = [
        "CASEID", "AIDSCASE", "TAIDS", "AIDSREPDyy", "DATE1yy", "DEATH", "DTHDATEyy",
        "macs_aids_documented", "macs_aids_status", "macs_aids_year", "macs_death_type",
        "macs_death_year", "macs_outcome_record_present",
    ]
    macs_out = macs[macs_cols].copy()
    if macs_out["CASEID"].duplicated().any():
        raise ValueError("MACS outcome extract has duplicate CASEID values; resolve before merging.")

    # WIHS: event-level records; preserve source rows and aggregate only to participant-level
    # event evidence. Do not infer absence of an event merely because a field is blank.
    wihs = pd.read_csv(WIHS_OUTCOME_FILE, dtype={"CASEID": "string", "VISIT": "string"}, low_memory=False)
    require_columns(
        wihs,
        ["CASEID", "VISIT", "OUTCAIDS", "OUTCDTH", "EVNTSTAT", "EVNTCODE",
         "TYPECONF", "DATEOUTCY", "DTHCLASS"],
        "WIHS outcome file",
    )
    wihs["CASEID"] = clean_id(wihs["CASEID"])
    wihs["VISIT"] = clean_id(wihs["VISIT"])
    for c in ["OUTCAIDS", "OUTCDTH", "EVNTSTAT", "EVNTCODE", "TYPECONF", "DATEOUTCY", "DTHCLASS"]:
        wihs[c] = numeric(wihs[c])

    # Codebook: OUTCAIDS/OUTCDTH 1=yes, 0=no; EVNTSTAT 1=prevalent, 2=incident,
    # -9=cannot be determined. Build flags only from explicit source values.
    wihs["wihs_aids_yes_record"] = wihs["OUTCAIDS"].eq(1)
    wihs["wihs_death_yes_record"] = wihs["OUTCDTH"].eq(1)
    wihs["wihs_incident_aids_record"] = wihs["OUTCAIDS"].eq(1) & wihs["EVNTSTAT"].eq(2)
    wihs["wihs_incident_death_record"] = wihs["OUTCDTH"].eq(1) & wihs["EVNTSTAT"].eq(2)
    wihs["wihs_prevalent_aids_record"] = wihs["OUTCAIDS"].eq(1) & wihs["EVNTSTAT"].eq(1)

    # Participant-level summary is evidence of observed records, not a complete event-free follow-up label.
    wihs_part = (
        wihs.groupby("CASEID", dropna=False)
        .agg(
            wihs_outcome_records=("CASEID", "size"),
            wihs_aids_ever_recorded=("wihs_aids_yes_record", "max"),
            wihs_death_ever_recorded=("wihs_death_yes_record", "max"),
            wihs_incident_aids_recorded=("wihs_incident_aids_record", "max"),
            wihs_incident_death_recorded=("wihs_incident_death_record", "max"),
            wihs_prevalent_aids_recorded=("wihs_prevalent_aids_record", "max"),
            wihs_first_event_year=("DATEOUTCY", "min"),
        )
        .reset_index()
    )
    wihs_part["wihs_outcome_record_present"] = True

    # Keep a concise event-level table for exact participant+visit joins, but do not use
    # it to overwrite the clinical visit table (multiple events may exist at a visit).
    wihs_visit = (
        wihs.groupby(["CASEID", "VISIT"], dropna=False)
        .agg(
            wihs_event_rows=("CASEID", "size"),
            wihs_visit_aids_yes=("wihs_aids_yes_record", "max"),
            wihs_visit_death_yes=("wihs_death_yes_record", "max"),
            wihs_visit_incident_aids=("wihs_incident_aids_record", "max"),
            wihs_visit_incident_death=("wihs_incident_death_record", "max"),
            wihs_visit_prevalent_aids=("wihs_prevalent_aids_record", "max"),
            wihs_visit_event_year=("DATEOUTCY", "min"),
            wihs_visit_confirmation_max=("TYPECONF", "max"),
        )
        .reset_index()
    )

    # Cohort-specific outcome merges. MACS has participant-level outcome data; WIHS has
    # participant-level evidence plus exact visit-level evidence.
    macs_base = base.loc[base["cohort"].eq("MACS")].merge(
        macs_out, on="CASEID", how="left", validate="many_to_one"
    )
    wihs_base = base.loc[base["cohort"].eq("WIHS")].merge(
        wihs_part, on="CASEID", how="left", validate="many_to_one"
    )
    wihs_base = wihs_base.merge(
        wihs_visit, on=["CASEID", "VISIT"], how="left", validate="one_to_one"
    )

    # Shared, explicit status fields; not a model target.
    macs_base["documented_aids_or_death_evidence"] = macs_base["macs_aids_documented"].eq(1)
    macs_base["outcome_evidence_status"] = macs_base["macs_aids_status"].fillna("outcome_unavailable")
    macs_base["outcome_source"] = "MACS outcome extract"

    wihs_base["documented_aids_or_death_evidence"] = (
        wihs_base["wihs_aids_ever_recorded"].eq(True)
        | wihs_base["wihs_death_ever_recorded"].eq(True)
    )
    wihs_base["outcome_evidence_status"] = "outcome_record_present_target_not_yet_assigned"
    wihs_base.loc[wihs_base["wihs_outcome_record_present"].isna(), "outcome_evidence_status"] = "outcome_unavailable"
    wihs_base["outcome_source"] = "WIHS event outcome file"

    combined = pd.concat([macs_base, wihs_base], ignore_index=True, sort=False)
    if len(combined) != len(base):
        raise AssertionError(f"Row count changed after outcome merge: {len(base)} -> {len(combined)}")
    if combined.duplicated(["participant_id", "VISIT"]).any():
        raise AssertionError("Duplicate participant-visit rows introduced by outcome merge.")

    # Explicitly prevent accidental interpretation as a final supervised target.
    combined["ml_target"] = pd.NA
    combined["ml_target_status"] = "not_assigned_requires_prediction_horizon_and_followup_rules"
    combined["outcome_definition_note"] = (
        "Documented AIDS/death outcome evidence; not confirmed ART treatment failure. "
        "Visit-level future target not assigned."
    )

    out_file = DATA / "hiv_longitudinal_outcomes_enriched.csv"
    combined.to_csv(out_file, index=False)

    summary = (
        combined.groupby("cohort", dropna=False)
        .agg(
            rows=("participant_id", "size"),
            participants=("participant_id", "nunique"),
            outcome_records_matched=("outcome_source", lambda s: s.notna().sum()),
            rows_with_documented_aids_or_death_evidence=("documented_aids_or_death_evidence", "sum"),
            rows_ml_target_unassigned=("ml_target_status", lambda s: s.eq("not_assigned_requires_prediction_horizon_and_followup_rules").sum()),
        )
        .reset_index()
    )
    summary_file = REPORTS / "outcome_merge_validation_by_cohort.csv"
    summary.to_csv(summary_file, index=False)

    report = [
        "# Outcome merge validation",
        "",
        f"Input dataset: {BASE_FILE.relative_to(ROOT)}",
        f"Output dataset: {out_file.relative_to(ROOT)}",
        "",
        f"- Input rows: {len(base):,}",
        f"- Output rows: {len(combined):,}",
        f"- Unique participants: {combined['participant_id'].nunique():,}",
        f"- Duplicate participant-visits after merge: {combined.duplicated(['participant_id', 'VISIT']).sum():,}",
        f"- Missing participant IDs: {combined['participant_id'].isna().sum():,}",
        "",
        "## Cohort summary",
        summary.to_string(index=False),
        "",
        "## Interpretation safeguards",
        "- MACS AIDSCASE codes are retained; code 1 means no CDC AIDS, 2 means CDC AIDS diagnosis, and 3 means AIDS by death only.",
        "- WIHS outcome fields are retained as event evidence; OUTCAIDS/OUTCDTH values are not treated as ART treatment failure.",
        "- Missing outcomes are not automatically treated as negative outcomes.",
        "- No visit-level ML target is assigned by this script.",
        "- Define an incident-event prediction horizon and follow-up/censoring rules before supervised training.",
        "- Existing input files were not modified.",
    ]
    report_file = REPORTS / "outcome_merge_validation.txt"
    report_file.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("Outcome enrichment complete.")
    print(f"Saved dataset: {out_file.relative_to(ROOT)}")
    print(f"Saved validation summary: {summary_file.relative_to(ROOT)}")
    print(f"Saved report: {report_file.relative_to(ROOT)}")
    print(f"Rows preserved: {len(base):,} -> {len(combined):,}")
    print(f"Unique participants: {combined['participant_id'].nunique():,}")
    print(f"Duplicate participant-visits: {combined.duplicated(['participant_id', 'VISIT']).sum():,}")
    print("\nCohort summary:")
    print(summary.to_string(index=False))
    print("\nNo ML target was assigned. This is intentional until the prediction horizon is defined.")

if __name__ == "__main__":
    main()
