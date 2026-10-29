
import joblib
import numpy as np
import pandas as pd
import shap

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)

VALID_INPUT = {
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


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_endpoint():
    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["model"] == "Gradient Boosting"
    assert 0 <= result["probability"] <= 1
    assert 0 <= result["class_probabilities"]["0"] <= 1
    assert 0 <= result["class_probabilities"]["1"] <= 1

    total = (
        result["class_probabilities"]["0"]
        + result["class_probabilities"]["1"]
    )
    assert abs(total - 1.0) < 1e-6


def test_explanation_endpoint():
    response = client.post("/explanation", json=VALID_INPUT)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["explanation_method"] == "SHAP TreeExplainer"
    assert len(result["top_features"]) > 0

    for feature in result["top_features"]:
        assert "feature" in feature
        assert "shap_value" in feature
        assert "absolute_shap_value" in feature


def test_missing_required_field():
    invalid_input = VALID_INPUT.copy()
    invalid_input.pop("CD4")

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


def test_invalid_numeric_field():
    invalid_input = VALID_INPUT.copy()
    invalid_input["CD4"] = "not-a-number"

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


def test_prediction_is_consistent_across_requests():
    first_response = client.post("/predict", json=VALID_INPUT)
    second_response = client.post("/predict", json=VALID_INPUT)

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first = first_response.json()
    second = second_response.json()

    assert first["prediction"] == second["prediction"]
    assert abs(first["probability"] - second["probability"]) < 1e-6
    assert first["class_probabilities"] == second["class_probabilities"]


def test_explanation_prediction_matches_prediction_endpoint():
    prediction_response = client.post("/predict", json=VALID_INPUT)
    explanation_response = client.post("/explanation", json=VALID_INPUT)

    assert prediction_response.status_code == 200
    assert explanation_response.status_code == 200

    prediction = prediction_response.json()
    explanation = explanation_response.json()

    assert explanation["prediction"] == prediction["prediction"]


def test_shap_values_reconstruct_model_raw_score():
    artifact = joblib.load(
        "models/gradient_boosting_treatment_proxy.joblib"
    )
    pipeline = (
        artifact["pipeline"]
        if isinstance(artifact, dict)
        else artifact
    )

    input_df = pd.DataFrame([VALID_INPUT])

    transformed = pipeline[:-1].transform(input_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    else:
        transformed = np.asarray(transformed)

    classifier = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(classifier)
    shap_values = np.asarray(explainer.shap_values(transformed))

    assert shap_values.ndim == 2
    assert shap_values.shape[0] == 1
    assert shap_values.shape[1] == transformed.shape[1]

    expected_value = float(
        np.asarray(explainer.expected_value).reshape(-1)[0]
    )
    shap_sum = float(shap_values[0].sum())

    reconstructed_score = expected_value + shap_sum
    actual_score = float(
        classifier.decision_function(transformed)[0]
    )

    assert abs(reconstructed_score - actual_score) < 1e-5


def test_inconsistent_cd4_change_is_rejected():
    invalid_input = VALID_INPUT.copy()
    invalid_input["CD4_change"] = 100

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


def test_inconsistent_viral_load_change_is_rejected():
    invalid_input = VALID_INPUT.copy()
    invalid_input["viral_load_change"] = -100

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422

def test_api_matches_saved_pipeline_prediction():
    artifact = joblib.load(
        "models/gradient_boosting_treatment_proxy.joblib"
    )
    pipeline = (
        artifact["pipeline"]
        if isinstance(artifact, dict)
        else artifact
    )

    input_df = pd.DataFrame([VALID_INPUT])
    direct_prediction = int(pipeline.predict(input_df)[0])
    direct_probabilities = pipeline.predict_proba(input_df)[0]

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200
    result = response.json()

    assert result["prediction"] == direct_prediction
    assert abs(
        result["class_probabilities"]["0"] - direct_probabilities[0]
    ) < 1e-6
    assert abs(
        result["class_probabilities"]["1"] - direct_probabilities[1]
    ) < 1e-6
