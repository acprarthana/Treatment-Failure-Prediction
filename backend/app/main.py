
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, model_validator


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_treatment_proxy.joblib"

app = FastAPI(
    title="HIV Treatment Outcome Prediction API",
    description=(
        "Prediction and SHAP-based explanations for a treatment-outcome "
        "proxy. Outputs are research predictions, not clinical diagnoses."
    ),
    version="1.0.0",
)

MODEL_ARTIFACT: dict[str, Any] | None = None
MODEL_ERROR: str | None = None


# --------------------------------------------------
# Request schema and validation
# --------------------------------------------------

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    CD4: float
    viral_load: float
    year: float
    CD4_previous: float
    viral_load_previous: float
    CD4_change: float
    viral_load_change: float
    years_since_previous_visit: float
    sex: str
    VISIT: float

    @model_validator(mode="after")
    def validate_values(self):
        numeric_fields = {
            "CD4": self.CD4,
            "viral_load": self.viral_load,
            "year": self.year,
            "CD4_previous": self.CD4_previous,
            "viral_load_previous": self.viral_load_previous,
            "CD4_change": self.CD4_change,
            "viral_load_change": self.viral_load_change,
            "years_since_previous_visit": self.years_since_previous_visit,
            "VISIT": self.VISIT,
        }

        for field_name, value in numeric_fields.items():
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be a finite number.")

        if self.CD4 < 0 or self.CD4_previous < 0:
            raise ValueError("CD4 values cannot be negative.")

        if self.viral_load < 0 or self.viral_load_previous < 0:
            raise ValueError("Viral load values cannot be negative.")

        if self.years_since_previous_visit < 0:
            raise ValueError(
                "years_since_previous_visit cannot be negative."
            )

        if not self.sex.strip():
            raise ValueError("sex cannot be empty.")

        # Change = current measurement - previous measurement.
        tolerance = 1e-6

        expected_cd4_change = self.CD4 - self.CD4_previous
        expected_viral_load_change = (
            self.viral_load - self.viral_load_previous
        )

        if abs(self.CD4_change - expected_cd4_change) > tolerance:
            raise ValueError(
                "CD4_change must equal CD4 - CD4_previous."
            )

        if (
            abs(self.viral_load_change - expected_viral_load_change)
            > tolerance
        ):
            raise ValueError(
                "viral_load_change must equal "
                "viral_load - viral_load_previous."
            )

        return self


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model_artifact() -> dict[str, Any]:
    """Load the saved model artifact and validate its structure."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if not isinstance(artifact, dict):
        raise ValueError(
            "Expected the saved model artifact to be a dictionary."
        )

    required_keys = {"pipeline", "features"}
    missing_keys = required_keys - set(artifact.keys())

    if missing_keys:
        raise ValueError(
            "Model artifact is missing required keys: "
            + ", ".join(sorted(missing_keys))
        )

    pipeline = artifact["pipeline"]

    if not hasattr(pipeline, "predict"):
        raise ValueError("The saved pipeline does not provide predict().")

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError(
            "The saved pipeline does not provide predict_proba()."
        )

    if not hasattr(pipeline, "named_steps"):
        raise ValueError("The saved pipeline does not expose named_steps.")

    if "preprocess" not in pipeline.named_steps:
        raise ValueError("The saved pipeline has no 'preprocess' step.")

    if "model" not in pipeline.named_steps:
        raise ValueError("The saved pipeline has no 'model' step.")

    if not isinstance(artifact["features"], (list, tuple)):
        raise ValueError("The model feature list must be a list or tuple.")

    if not artifact["features"]:
        raise ValueError("The model feature list is empty.")

    return artifact


def get_model_artifact() -> dict[str, Any]:
    """Return the cached model artifact, loading it if necessary."""
    global MODEL_ARTIFACT, MODEL_ERROR

    if MODEL_ARTIFACT is not None:
        return MODEL_ARTIFACT

    try:
        MODEL_ARTIFACT = load_model_artifact()
        MODEL_ERROR = None
        return MODEL_ARTIFACT
    except Exception as exc:
        MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable.",
        ) from exc


def get_pipeline():
    """Return the trained sklearn pipeline."""
    return get_model_artifact()["pipeline"]


@app.on_event("startup")
def startup_load_model():
    """Attempt to load the model at application startup."""
    global MODEL_ARTIFACT, MODEL_ERROR

    try:
        MODEL_ARTIFACT = load_model_artifact()
        MODEL_ERROR = None
    except Exception as exc:
        MODEL_ARTIFACT = None
        MODEL_ERROR = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------
# Input preparation and model helpers
# --------------------------------------------------

def request_to_dataframe(
    request: PredictionRequest,
) -> pd.DataFrame:
    """Convert validated request data into the saved model's feature order."""
    artifact = get_model_artifact()
    feature_columns = list(artifact["features"])
    request_data = request.model_dump()

    missing_features = [
        feature for feature in feature_columns
        if feature not in request_data
    ]

    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=(
                "API request schema is missing model features: "
                + ", ".join(missing_features)
            ),
        )

    return pd.DataFrame(
        [[request_data[feature] for feature in feature_columns]],
        columns=feature_columns,
    )


def get_model_name(artifact: dict[str, Any]) -> str:
    return str(artifact.get("model_name", "Gradient Boosting"))


def json_safe_number(value: Any) -> float | None:
    """Convert a numeric value to a JSON-safe Python float."""
    number = float(value)
    return number if np.isfinite(number) else None


def predict_with_pipeline(input_df: pd.DataFrame) -> dict[str, Any]:
    """Return the predicted class and class probabilities."""
    artifact = get_model_artifact()
    pipeline = artifact["pipeline"]

    prediction = int(pipeline.predict(input_df)[0])
    probabilities = pipeline.predict_proba(input_df)[0]
    classes = pipeline.named_steps["model"].classes_

    probability_by_class = {
        str(label): json_safe_number(probability)
        for label, probability in zip(classes, probabilities)
    }

    return {
        "prediction": prediction,
        "probabilities": probability_by_class,
        "class_probabilities": probability_by_class,
        "model_name": get_model_name(artifact),
    }


# --------------------------------------------------
# SHAP helpers
# --------------------------------------------------

def get_transformed_feature_names(
    pipeline,
    transformed_data,
    fallback_names: list[str],
) -> list[str]:
    """Get feature names after preprocessing."""
    preprocessor = pipeline.named_steps["preprocess"]

    try:
        names = list(preprocessor.get_feature_names_out())
        if len(names) == transformed_data.shape[1]:
            return names
    except Exception:
        pass

    if len(fallback_names) == transformed_data.shape[1]:
        return fallback_names

    return [
        f"feature_{index}"
        for index in range(transformed_data.shape[1])
    ]


def make_feature_description(feature_name: str) -> str:
    """Convert common transformed feature names into readable descriptions."""
    readable_name = feature_name

    if "__" in readable_name:
        readable_name = readable_name.split("__", 1)[1]

    replacements = {
        "CD4_previous": "Previous CD4 count",
        "viral_load_previous": "Previous viral load",
        "CD4_change": "Change in CD4 count",
        "viral_load_change": "Change in viral load",
        "years_since_previous_visit": "Years since previous visit",
        "CD4": "Current CD4 count",
        "viral_load": "Current viral load",
        "year": "Year of observation",
        "VISIT": "Visit identifier",
        "sex": "Sex",
    }

    return replacements.get(
        readable_name,
        readable_name.replace("_", " "),
    )


def calculate_shap_explanation(
    input_df: pd.DataFrame,
    predicted_class: int,
) -> dict[str, Any]:
    """
    Calculate SHAP values for the saved tree-based classifier.

    For the current binary GradientBoostingClassifier, SHAP values
    explain the class 1 raw model score, not probability changes.
    """
    pipeline = get_pipeline()
    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["model"]

    transformed = preprocessor.transform(input_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    feature_names = get_transformed_feature_names(
        pipeline,
        transformed,
        list(input_df.columns),
    )

    explainer = shap.TreeExplainer(classifier)
    shap_output = explainer.shap_values(transformed)

    if isinstance(shap_output, list):
        class_index = 1 if len(shap_output) > 1 else 0
        shap_values = np.asarray(shap_output[class_index])[0]
    else:
        shap_array = np.asarray(shap_output)

        if shap_array.ndim == 3:
            if shap_array.shape[-1] > 1:
                shap_values = shap_array[0, :, 1]
            else:
                shap_values = shap_array[0, :, 0]
        elif shap_array.ndim == 2:
            shap_values = shap_array[0]
        else:
            raise ValueError(
                f"Unexpected SHAP value shape: {shap_array.shape}"
            )

    expected_value = np.asarray(explainer.expected_value).reshape(-1)
    base_value = float(
        expected_value[1]
        if expected_value.size > 1
        else expected_value[0]
    )

    if len(shap_values) != len(feature_names):
        raise ValueError(
            "SHAP values do not match the transformed feature count."
        )

    features = []

    for name, value in zip(feature_names, shap_values):
        shap_value = float(value)

        if not np.isfinite(shap_value):
            raise ValueError(
                "SHAP returned a non-finite feature contribution."
            )

        features.append(
            {
                "feature": name,
                "description": make_feature_description(name),
                "shap_value": shap_value,
                "absolute_shap_value": abs(shap_value),
                "direction": (
                    "positive contribution to class 1 raw score"
                    if shap_value > 0
                    else (
                        "negative contribution to class 1 raw score"
                        if shap_value < 0
                        else "no contribution to class 1 raw score"
                    )
                ),
            }
        )

    features.sort(
        key=lambda item: item["absolute_shap_value"],
        reverse=True,
    )

    shap_sum = float(np.sum(shap_values))

    return {
        "method": "SHAP TreeExplainer",
        "explained_output": (
            "Class 1 raw model score; SHAP values are not probability changes."
        ),
        "expected_value": base_value,
        "shap_value_sum": shap_sum,
        "reconstructed_raw_score": base_value + shap_sum,
        "features": features,
        "top_features": features[:10],
    }


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/health")
def health_check():
    """Report API status and model availability."""
    global MODEL_ARTIFACT, MODEL_ERROR

    if MODEL_ARTIFACT is None:
        try:
            MODEL_ARTIFACT = load_model_artifact()
            MODEL_ERROR = None
        except Exception as exc:
            MODEL_ERROR = f"{type(exc).__name__}: {exc}"

    return {
        "status": "ok" if MODEL_ARTIFACT is not None else "degraded",
        "service": "HIV Treatment Outcome Prediction API",
        "model_loaded": MODEL_ARTIFACT is not None,
        "model_error": MODEL_ERROR,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    """Predict the project's treatment-outcome proxy."""
    try:
        input_df = request_to_dataframe(request)
        result = predict_with_pipeline(input_df)

        result["model"] = result["model_name"]
        result["probability"] = result["class_probabilities"].get(
            str(result["prediction"])
        )

        result["target_note"] = (
            "The model predicts the project's treatment-outcome proxy, "
            "not independently confirmed clinical HIV treatment failure."
        )

        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal processing error.",
        ) from exc


@app.post("/explanation")
def explain_prediction(request: PredictionRequest):
    """Return a prediction and its largest SHAP contributions."""
    try:
        input_df = request_to_dataframe(request)
        prediction_result = predict_with_pipeline(input_df)

        explanation = calculate_shap_explanation(
            input_df,
            prediction_result["prediction"],
        )

        return {
            **prediction_result,
            "explanation_method": explanation["method"],
            "top_features": explanation["top_features"],
            "explanation": explanation,
            "target_note": (
                "This explanation describes model behavior for the project's "
                "treatment-outcome proxy. It does not establish causation "
                "or provide a clinical diagnosis."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Explanation failed due to an internal processing error.",
        ) from exc
