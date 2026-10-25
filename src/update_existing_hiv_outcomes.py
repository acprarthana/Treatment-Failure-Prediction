"""
Update the EXISTING HIV outcome-enriched dataset in place.
No second dataset is created. The existing enriched CSV is backed up once before overwrite.

Adds a visit-level MACS treatment-outcome proxy from medication stop reasons:
  1 = code 65 appears in any STOT1/STOT2/STOT3/STOTH field
  0 = at least one stop reason is recorded, and none is code 65
  missing = no interpretable stop reason at that participant-visit

This is a reported failure-related medication-stop proxy, NOT clinically confirmed ART failure.
AIDS/death variables remain separate clinical outcomes and are not relabeled as treatment failure.
Duplicate medication records are collapsed to one CASEID+VISIT summary before merging.
MACS IDs are normalized for leading-zero differences before outcome merging.
"""

from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

BASE_FILE = DATA / "combined_ml_dataset_provisional.csv"
EXISTING_ENRICHED = DATA / "hiv_longitudinal_outcomes_enriched.csv"
MACS_OUTCOME_FILE = DATA / "outcome_selected.csv"
MACS_MED_FILE = DATA / "macs_drugf1_selected_v2.csv"
WIHS_OUTCOME_FILE = RAW / "WIHS" / "WIHS PDS" / "CSV" / "outcome.csv"


def clean_id(s):
    s = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return s.replace({"": pd.NA, ".": pd.NA, "nan": pd.NA, "None": pd.NA})


def canonical_id(s):
    """Canonical numeric participant ID: remove leading zeroes without losing nonnumeric IDs."""
    s = clean_id(s)
    is_digits = s.str.fullmatch(r"\d+", na=False)
    out = s.copy()
    out.loc[is_digits] = out.loc[is_digits].str.lstrip("0").replace("", "0")
    return out


def numeric(s):
    return pd.to_numeric(
        s.replace({".": pd.NA, "": pd.NA, "-9": pd.NA, "-1": pd.NA}),
        errors="coerce",
    )


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def read_csv(path, **kwargs):
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path, low_memory=False, **kwargs)


def build_macs_treatment_proxy():
    med = read_csv(MACS_MED_FILE, dtype={"CASEID": "string", "VISIT": "string"})
    require_columns(med, ["CASEID", "VISIT", "STOT1", "STOT2", "STOT3", "STOTH"], "MACS medication file")
    med["CASEID"] = canonical_id(med["CASEID"])
    med["VISIT"] = clean_id(med["VISIT"])
    reason_cols = ["STOT1", "STOT2", "STOT3", "STOTH"]
    reason_values = med[reason_cols].astype("string").apply(
        lambda col: col.str.strip().replace({".": pd.NA, "": pd.NA, "nan": pd.NA})
    )
    has_any_reason = reason_values.notna().any(axis=1)
    has_code_65 = reason_values.eq("65").any(axis=1)
    med["_has_reason"] = has_any_reason
    med["_code65"] = has_code_65

    # Collapse repeated medication records at the same participant-visit.
    grouped = med.groupby(["CASEID", "VISIT"], dropna=False).agg(
        medication_record_count=("CASEID", "size"),
        code65_record_count=("_code65", "sum"),
        any_stop_reason_recorded=("_has_reason", "max"),
        reported_failure_signal=("_code65", "max"),
    ).reset_index()

    grouped["treatment_outcome"] = pd.Series(pd.NA, index=grouped.index, dtype="Int64")
    known = grouped["any_stop_reason_recorded"].eq(True)
    grouped.loc[known & grouped["reported_failure_signal"].eq(True), "treatment_outcome"] = 1
    grouped.loc[known & grouped["reported_failure_signal"].eq(False), "treatment_outcome"] = 0
    grouped["treatment_outcome_status"] = "unknown_no_interpretable_stop_reason"
    grouped.loc[grouped["treatment_outcome"].eq(1), "treatment_outcome_status"] = "proxy_failure_related_stop_reason_code_65"
    grouped.loc[grouped["treatment_outcome"].eq(0), "treatment_outcome_status"] = "documented_other_stop_reason_no_code_65"
    grouped["treatment_outcome_source"] = "MACS medication stop reason fields STOT1/STOT2/STOT3/STOTH"
    grouped["treatment_outcome_definition"] = (
        "Proxy only: code 65 is failure-related medication stop reason; not confirmed ART treatment failure."
    )
    return grouped


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Start from the established provisional visit table and update the established enriched file in place.
    base = read_csv(BASE_FILE, dtype={"CASEID": "string", "VISIT": "string"})
    require_columns(base, ["CASEID", "VISIT", "cohort", "participant_id"], "Combined dataset")
    base["CASEID"] = canonical_id(base["CASEID"])
    base["VISIT"] = clean_id(base["VISIT"])
    base["cohort"] = base["cohort"].astype("string").str.strip().str.upper()

    if base.duplicated(["participant_id", "VISIT"]).any():
        raise ValueError("Base dataset already has duplicate participant_id+VISIT rows; stopping to prevent redundancy.")

    # Correct MACS participant-level outcomes merge using canonical IDs.
    macs = read_csv(MACS_OUTCOME_FILE, dtype={"CASEID": "string"})
    require_columns(macs, ["CASEID", "AIDSCASE", "TAIDS", "AIDSREPDyy", "DATE1yy", "DEATH", "DTHDATEyy"], "MACS outcome extract")
    macs["CASEID"] = canonical_id(macs["CASEID"])
    for c in ["AIDSCASE", "TAIDS", "AIDSREPDyy", "DATE1yy", "DEATH", "DTHDATEyy"]:
        macs[c] = numeric(macs[c])
    if macs["CASEID"].duplicated().any():
        # A participant-level outcome file should have one record per participant.
        macs = macs.sort_values("CASEID").drop_duplicates("CASEID", keep="first")

    macs["macs_aids_documented"] = macs["AIDSCASE"].map({1: 0, 2: 1, 3: 1}).astype("Int64")
    macs["macs_aids_status"] = macs["AIDSCASE"].map({
        1: "no_CDC_AIDS_documented", 2: "CDC_AIDS_diagnosis", 3: "AIDS_by_death_only"
    }).fillna("unknown")
    macs["macs_aids_year"] = macs["DATE1yy"].combine_first(macs["AIDSREPDyy"])
    macs["macs_death_type"] = macs["DEATH"]
    macs["macs_death_year"] = macs["DTHDATEyy"]
    macs["macs_outcome_record_present"] = True
    macs_out = macs[[
        "CASEID", "AIDSCASE", "TAIDS", "AIDSREPDyy", "DATE1yy", "DEATH", "DTHDATEyy",
        "macs_aids_documented", "macs_aids_status", "macs_aids_year",
        "macs_death_type", "macs_death_year", "macs_outcome_record_present"
    ]].copy()

    macs_base = base.loc[base["cohort"].eq("MACS")].merge(
        macs_out, on="CASEID", how="left", validate="many_to_one", suffixes=("", "_source")
    )
    macs_base["documented_aids_or_death_evidence"] = macs_base["macs_aids_documented"].eq(1)
    macs_base["outcome_evidence_status"] = macs_base["macs_aids_status"].fillna("outcome_unavailable")
    macs_base["outcome_source"] = macs_base["macs_outcome_record_present"].map(
        {True: "MACS outcome extract matched", False: "MACS outcome extract not matched"}
    ).fillna("MACS outcome extract not matched")

    # WIHS outcomes: retain explicit event evidence; no inference of treatment failure.
    wihs = read_csv(WIHS_OUTCOME_FILE, dtype={"CASEID": "string", "VISIT": "string"})
    require_columns(wihs, ["CASEID", "VISIT", "OUTCAIDS", "OUTCDTH", "EVNTSTAT", "EVNTCODE",
                            "TYPECONF", "DATEOUTCY", "DTHCLASS"], "WIHS outcome file")
    wihs["CASEID"] = canonical_id(wihs["CASEID"])
    wihs["VISIT"] = clean_id(wihs["VISIT"])
    for c in ["OUTCAIDS", "OUTCDTH", "EVNTSTAT", "EVNTCODE", "TYPECONF", "DATEOUTCY", "DTHCLASS"]:
        wihs[c] = numeric(wihs[c])
    wihs["_aids_yes"] = wihs["OUTCAIDS"].eq(1)
    wihs["_death_yes"] = wihs["OUTCDTH"].eq(1)
    wihs["_incident_aids"] = wihs["_aids_yes"] & wihs["EVNTSTAT"].eq(2)
    wihs["_incident_death"] = wihs["_death_yes"] & wihs["EVNTSTAT"].eq(2)
    wihs["_prevalent_aids"] = wihs["_aids_yes"] & wihs["EVNTSTAT"].eq(1)

    wihs_part = wihs.groupby("CASEID", dropna=False).agg(
        wihs_outcome_records=("CASEID", "size"),
        wihs_aids_ever_recorded=("_aids_yes", "max"),
        wihs_death_ever_recorded=("_death_yes", "max"),
        wihs_incident_aids_recorded=("_incident_aids", "max"),
        wihs_incident_death_recorded=("_incident_death", "max"),
        wihs_prevalent_aids_recorded=("_prevalent_aids", "max"),
        wihs_first_event_year=("DATEOUTCY", "min"),
    ).reset_index()
    wihs_part["wihs_outcome_record_present"] = True

    wihs_visit = wihs.groupby(["CASEID", "VISIT"], dropna=False).agg(
        wihs_event_rows=("CASEID", "size"),
        wihs_visit_aids_yes=("_aids_yes", "max"),
        wihs_visit_death_yes=("_death_yes", "max"),
        wihs_visit_incident_aids=("_incident_aids", "max"),
        wihs_visit_incident_death=("_incident_death", "max"),
        wihs_visit_prevalent_aids=("_prevalent_aids", "max"),
        wihs_visit_event_year=("DATEOUTCY", "min"),
        wihs_visit_confirmation_max=("TYPECONF", "max"),
    ).reset_index()

    wihs_base = base.loc[base["cohort"].eq("WIHS")].merge(
        wihs_part, on="CASEID", how="left", validate="many_to_one"
    ).merge(wihs_visit, on=["CASEID", "VISIT"], how="left", validate="one_to_one")
    wihs_base["documented_aids_or_death_evidence"] = (
        wihs_base["wihs_aids_ever_recorded"].eq(True) | wihs_base["wihs_death_ever_recorded"].eq(True)
    )
    wihs_base["outcome_evidence_status"] = "outcome_record_present_target_not_yet_assigned"
    wihs_base.loc[wihs_base["wihs_outcome_record_present"].isna(), "outcome_evidence_status"] = "outcome_unavailable"
    wihs_base["outcome_source"] = wihs_base["wihs_outcome_record_present"].map(
        {True: "WIHS event outcome file matched", False: "WIHS event outcome file not matched"}
    ).fillna("WIHS event outcome file not matched")

    # Merge the treatment proxy only for MACS and only after collapsing medication records.
    proxy = build_macs_treatment_proxy()
    # Remove stale proxy fields inherited from the provisional dataset.
    stale_proxy_cols = [
        "reported_failure_signal",
        "reported_failure_signal_x",
        "reported_failure_signal_y",
        "treatment_outcome",
        "treatment_outcome_status",
        "treatment_outcome_source",
        "treatment_outcome_definition",
        "medication_record_count",
        "code65_record_count",
        "any_stop_reason_recorded",
    ]
    macs_base = macs_base.drop(
        columns=[c for c in stale_proxy_cols if c in macs_base.columns],
        errors="ignore",
    )

    macs_base = macs_base.merge(
        proxy,
        on=["CASEID", "VISIT"],
        how="left",
        validate="one_to_one",
    )
    # WIHS has no verified comparable stop-reason code mapping; keep target unknown rather than fabricate.
    wihs_base["treatment_outcome"] = pd.Series(pd.NA, index=wihs_base.index, dtype="Int64")
    wihs_base["treatment_outcome_status"] = "not_available_comparable_definition_not_verified"
    wihs_base["treatment_outcome_source"] = "not assigned: WIHS stop-reason code mapping not verified"
    wihs_base["treatment_outcome_definition"] = (
        "No comparable WIHS treatment-failure label assigned because stop-reason mapping is unverified."
    )
    for c in ["medication_record_count", "code65_record_count", "any_stop_reason_recorded", "reported_failure_signal"]:
        if c not in wihs_base.columns:
            wihs_base[c] = pd.NA

    combined = pd.concat([macs_base, wihs_base], ignore_index=True, sort=False)
    if len(combined) != len(base):
        raise AssertionError(f"Row count changed: {len(base):,} -> {len(combined):,}")
    if combined.duplicated(["participant_id", "VISIT"]).any():
        raise AssertionError("Duplicate participant-visit rows introduced; refusing to save.")

    # Do not retain stale target fields from earlier script runs.
    combined["ml_target"] = pd.NA
    combined["ml_target_status"] = "not_assigned: treatment proxy is not confirmed failure and prediction horizon is undefined"
    combined["outcome_definition_note"] = (
        "AIDS/death evidence is separate from treatment outcome. treatment_outcome=1 is MACS code-65 proxy, "
        "0 is a documented non-65 stop reason without code 65, missing is unknown/not comparable. "
        "This is not clinically confirmed ART treatment failure."
    )

    # One-time safety backup, then overwrite the SAME established enriched dataset.
    if EXISTING_ENRICHED.exists():
        backup = EXISTING_ENRICHED.with_suffix(".csv.bak")
        if not backup.exists():
            shutil.copy2(EXISTING_ENRICHED, backup)
    combined.to_csv(EXISTING_ENRICHED, index=False)

    macs_summary = macs_base[["macs_outcome_record_present"]].notna().sum().iloc[0]
    macs_matched = int(macs_base["macs_outcome_record_present"].eq(True).sum())
    summary = combined.groupby("cohort", dropna=False).agg(
        rows=("participant_id", "size"),
        participants=("participant_id", "nunique"),
        treatment_outcome_positive=("treatment_outcome", lambda s: s.eq(1).sum()),
        treatment_outcome_negative=("treatment_outcome", lambda s: s.eq(0).sum()),
        treatment_outcome_unknown=("treatment_outcome", lambda s: s.isna().sum()),
        documented_aids_death_evidence=("documented_aids_or_death_evidence", "sum"),
    ).reset_index()
    summary.to_csv(REPORTS / "outcome_merge_validation_by_cohort.csv", index=False)
    (REPORTS / "outcome_merge_validation.txt").write_text(
        "# Existing outcome dataset update\n\n"
        f"- Existing enriched file updated in place: {EXISTING_ENRICHED.relative_to(ROOT)}\n"
        f"- Rows preserved: {len(base):,} -> {len(combined):,}\n"
        f"- Unique participants: {combined['participant_id'].nunique():,}\n"
        f"- Duplicate participant-visits: {combined.duplicated(['participant_id', 'VISIT']).sum():,}\n"
        f"- MACS outcome records actually matched: {macs_matched:,} of {len(macs_base):,} MACS rows\n"
        f"- MACS medication participant-visits after de-duplication: {len(proxy):,}\n"
        f"- MACS code-65 proxy-positive participant-visits: {int(proxy['treatment_outcome'].eq(1).sum()):,}\n"
        "- AIDS/death evidence is not treated as ART treatment failure.\n"
        "- No new dataset file is created; existing enriched CSV is overwritten after a one-time .bak backup.\n\n"
        "## Summary by cohort\n\n" + summary.to_string(index=False) + "\n",
        encoding="utf-8"
    )
    print(f"Updated existing dataset in place: {EXISTING_ENRICHED.relative_to(ROOT)}")
    print(f"Rows preserved: {len(base):,}; duplicate participant-visits: {combined.duplicated(['participant_id', 'VISIT']).sum():,}")
    print(f"MACS outcome matched rows: {macs_matched:,}/{len(macs_base):,}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
