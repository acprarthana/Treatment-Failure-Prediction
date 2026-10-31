
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS visits (
    visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,

    visit_date TEXT NOT NULL,

    cd4 REAL NOT NULL,
    viral_load REAL NOT NULL,
    year INTEGER NOT NULL,
    cd4_previous REAL NOT NULL,
    viral_load_previous REAL NOT NULL,
    cd4_change REAL NOT NULL,
    viral_load_change REAL NOT NULL,
    years_since_previous_visit REAL NOT NULL,
    sex TEXT NOT NULL,
    visit_number INTEGER NOT NULL,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (patient_id)
        REFERENCES patients(patient_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id INTEGER NOT NULL UNIQUE,

    predicted_class INTEGER NOT NULL,
    probability REAL NOT NULL,
    class_probabilities TEXT NOT NULL,
    model_name TEXT NOT NULL,

    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (visit_id)
        REFERENCES visits(visit_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_visits_patient_id
ON visits(patient_id);

CREATE INDEX IF NOT EXISTS idx_predictions_visit_id
ON predictions(visit_id);
