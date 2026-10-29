
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
    "VISIT": 180
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


