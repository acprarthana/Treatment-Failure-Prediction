
import database.db as db


def test_database_operations(tmp_path, monkeypatch):
    # Use a fresh temporary database for every test run
    test_db_path = tmp_path / "test_hiv_prediction.db"
    monkeypatch.setattr(db, "DATABASE_PATH", test_db_path)

    db.initialize_database()

    patient_id = "TEST_PATIENT_001"

    clinical_data = {
        "CD4": 500,
        "viral_load": 1000,
        "year": 2018,
        "CD4_previous": 520,
        "viral_load_previous": 800,
        "CD4_change": -20,
        "viral_load_change": 200,
        "years_since_previous_visit": 1,
        "sex": "Male",
        "VISIT": 180,
    }

    db.add_patient(patient_id)

    visit_id = db.add_visit(
        patient_id,
        "2018-01-15",
        clinical_data
    )

    db.add_prediction(
        visit_id,
        0,
        0.85,
        {"0": 0.85, "1": 0.15},
        "Gradient Boosting"
    )

    history = db.get_patient_history(patient_id)

    assert len(history) == 1
    assert history[0]["patient_id"] == patient_id
    assert history[0]["predicted_class"] == 0
    assert history[0]["model_name"] == "Gradient Boosting"
