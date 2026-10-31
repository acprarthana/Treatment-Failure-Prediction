
import sqlite3
import json
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Database location
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "hiv_prediction.db"

# Schema file
SCHEMA_PATH = DATABASE_DIR / "schema.sql"


def get_connection():
    """Create and return a SQLite database connection."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_database():
    """Create database tables using schema.sql."""
    with get_connection() as connection:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        connection.executescript(schema)


def add_patient(patient_id):
    """Add a pseudonymous patient record."""
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO patients (patient_id)
            VALUES (?)
            """,
            (patient_id,)
        )


def add_visit(patient_id, visit_date, clinical_data):
    """Store one patient visit and return its database ID."""

    required_fields = [
        "CD4",
        "viral_load",
        "year",
        "CD4_previous",
        "viral_load_previous",
        "CD4_change",
        "viral_load_change",
        "years_since_previous_visit",
        "sex",
        "VISIT"
    ]

    missing_fields = [
        field for field in required_fields
        if field not in clinical_data
    ]

    if missing_fields:
        raise ValueError(
            f"Missing clinical fields: {missing_fields}"
        )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO visits (
                patient_id,
                visit_date,
                cd4,
                viral_load,
                year,
                cd4_previous,
                viral_load_previous,
                cd4_change,
                viral_load_change,
                years_since_previous_visit,
                sex,
                visit_number
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                visit_date,
                clinical_data["CD4"],
                clinical_data["viral_load"],
                clinical_data["year"],
                clinical_data["CD4_previous"],
                clinical_data["viral_load_previous"],
                clinical_data["CD4_change"],
                clinical_data["viral_load_change"],
                clinical_data["years_since_previous_visit"],
                clinical_data["sex"],
                clinical_data["VISIT"]
            )
        )

        return cursor.lastrowid


def add_prediction(
    visit_id,
    predicted_class,
    probability,
    class_probabilities,
    model_name
):
    """Store the prediction associated with a visit."""

    if predicted_class not in (0, 1):
        raise ValueError("predicted_class must be 0 or 1")

    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                visit_id,
                predicted_class,
                probability,
                class_probabilities,
                model_name
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                visit_id,
                predicted_class,
                probability,
                json.dumps(class_probabilities),
                model_name
            )
        )


def get_patient_history(patient_id):
    """Retrieve visits and predictions for one patient."""

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                visits.*,
                predictions.prediction_id,
                predictions.predicted_class,
                predictions.probability,
                predictions.class_probabilities,
                predictions.model_name,
                predictions.created_at AS prediction_created_at
            FROM visits
            LEFT JOIN predictions
                ON visits.visit_id = predictions.visit_id
            WHERE visits.patient_id = ?
            ORDER BY visits.visit_date DESC, visits.visit_id DESC
            """,
            (patient_id,)
        ).fetchall()

        results = [dict(row) for row in rows]

        for result in results:
            if result["class_probabilities"]:
                result["class_probabilities"] = json.loads(
                    result["class_probabilities"]
                )

        return results
