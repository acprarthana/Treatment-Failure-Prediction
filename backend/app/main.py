
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# Project paths and API configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "gradient_boosting_treatment_proxy.joblib"
)

app = FastAPI(
    title="HIV Treatment Failure Prediction API",
    description=(
        "Prediction and patient-specific SHAP explanations "
        "for the treatment-outcome proxy model."
    ),
    version="1.2.0",
)


FEATURE_COLUMNS = [
    "CD4",
    "viral_load",
    "year",
    "CD4_previous",
    "viral_load_previous",
    "CD4_change",
    "viral_load_change",
    "years_since_previous_visit",
    "sex",
    "VISIT",
]


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class PredictionRequest(BaseModel):
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


# --------------------------------------------------
# Load the saved model
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run src/save_final_model.py first."
        )

    saved_object = joblib.load(MODEL_PATH)

    # Support a saved dictionary or a directly saved pipeline.
    if isinstance(saved_object, dict):
        model_pipeline = saved_object.get("pipeline")

        if model_pipeline is None:
            raise ValueError(
                "The saved model dictionary does not contain "
                "a 'pipeline' entry."
            )
    else:
        model_pipeline = saved_object

    if not isinstance(model_pipeline, Pipeline):
        raise TypeError(
            "The saved model is not an sklearn Pipeline."
        )

    if "model" not in model_pipeline.named_steps:
        raise ValueError(
            "The pipeline does not contain a step named 'model'."
        )

    return model_pipeline


try:
    pipeline = load_model()
    model_load_error = None
except Exception as exc:
    pipeline = None
    model_load_error = str(exc)


def get_pipeline():
    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load prediction model: {model_load_error}",
        )

    return pipeline


# --------------------------------------------------
# Convert API request to model input
# --------------------------------------------------

def request_to_dataframe(
    request: PredictionRequest,
) -> pd.DataFrame:
    values = request.model_dump()

    return pd.DataFrame(
        [[values[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )


# --------------------------------------------------
# Get the transformed feature names for SHAP
# --------------------------------------------------

def get_transformed_features(model_pipeline, input_df):
    """Apply preprocessing without running the final prediction step."""

    preprocessing = model_pipeline[:-1]
    transformed = preprocessing.transform(input_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    try:
        feature_names = [
            str(name)
            for name in preprocessing.get_feature_names_out()
        ]
    except Exception:
        feature_names = [
            f"feature_{index}"
            for index in range(transformed.shape[1])
        ]

    if len(feature_names) != transformed.shape[1]:
        feature_names = [
            f"feature_{index}"
            for index in range(transformed.shape[1])
        ]

    return transformed, feature_names


# --------------------------------------------------
# Human-readable feature descriptions
# --------------------------------------------------

def describe_shap_feature(feature_name: str) -> dict:
    """Return a readable label and description for a model feature."""

    feature = feature_name.split("__", 1)[-1]

    descriptions = {
        "CD4": (
            "Current CD4 count",
            "The recorded CD4 count at the current visit.",
        ),
        "viral_load": (
            "Current viral load",
            "The recorded viral load at the current visit.",
        ),
        "year": (
            "Visit year",
            "The calendar year associated with this visit.",
        ),
        "CD4_previous": (
            "Previous CD4 count",
            "The recorded CD4 count at the previous visit.",
        ),
        "viral_load_previous": (
            "Previous viral load",
            "The recorded viral load at the previous visit.",
        ),
        "CD4_change": (
            "Change in CD4 count",
            "The change in CD4 count between visits.",
        ),
        "viral_load_change": (
            "Change in viral load",
            "The change in viral load between visits.",
        ),
        "years_since_previous_visit": (
            "Time since previous visit",
            "The elapsed time since the previous visit, measured in years.",
        ),
    }

    if feature in descriptions:
        label, description = descriptions[feature]

    elif feature.startswith("VISIT_"):
        category = feature.removeprefix("VISIT_")
        label = f"Visit category: {category}"
        description = (
            f"Indicates the recorded visit category {category}."
        )

    elif feature.startswith("sex_"):
        category = feature.removeprefix("sex_")
        label = f"Recorded sex: {category}"
        description = (
            f"Indicates the encoded sex category {category}."
        )

    else:
        label = feature.replace("_", " ").replace("-", " ").title()
        description = (
            "A transformed feature used by the prediction model."
        )

    return {
        "feature_label": label,
        "description": description,
    }


# --------------------------------------------------
# Patient-specific explanations
# --------------------------------------------------

def explain_patient_feature(
    feature_name: str,
    input_df: pd.DataFrame,
) -> dict:
    """
    Describe observed measurements and changes.

    These descriptions explain the patient's recorded data.
    SHAP values separately describe model contributions.
    Neither establishes causation.
    """

    row = input_df.iloc[0]
    feature = feature_name.split("__", 1)[-1]

    if feature == "CD4":
        current = float(row["CD4"])
        previous = float(row["CD4_previous"])
        change = current - previous

        if change > 0:
            change_text = f"increased by {change:g}"
        elif change < 0:
            change_text = f"decreased by {abs(change):g}"
        else:
            change_text = "did not change"

        return {
            "current_value": current,
            "previous_value": previous,
            "observed_change": change_text,
            "change_value": change,
            "patient_explanation": (
                f"The recorded CD4 count {change_text} between "
                f"visits, from {previous:g} to {current:g}. "
                "The SHAP value shows how the current CD4 feature "
                "contributed to the model output being explained. "
                "The decrease alone does not establish treatment failure."
            ),
        }

    if feature == "viral_load":
        current = float(row["viral_load"])
        previous = float(row["viral_load_previous"])
        change = current - previous

        if change > 0:
            change_text = f"increased by {change:g}"
        elif change < 0:
            change_text = f"decreased by {abs(change):g}"
        else:
            change_text = "did not change"

        return {
            "current_value": current,
            "previous_value": previous,
            "observed_change": change_text,
            "change_value": change,
            "patient_explanation": (
                f"The recorded viral load {change_text} between "
                f"visits, from {previous:g} to {current:g}. "
                "The SHAP value shows how the current viral-load "
                "feature contributed to the model output being explained. "
                "This change alone does not confirm treatment failure."
            ),
        }

    if feature == "CD4_previous":
        value = float(row["CD4_previous"])

        return {
            "current_value": value,
            "patient_explanation": (
                f"The previous recorded CD4 count was {value:g}. "
                "Its SHAP value indicates its contribution to the "
                "model output being explained."
            ),
        }

    if feature == "viral_load_previous":
        value = float(row["viral_load_previous"])

        return {
            "current_value": value,
            "patient_explanation": (
                f"The previous recorded viral load was {value:g}. "
                "Its SHAP value indicates its contribution to the "
                "model output being explained."
            ),
        }

    if feature == "CD4_change":
        value = float(row["CD4_change"])

        return {
            "current_value": value,
            "patient_explanation": (
                f"The supplied CD4-change feature is {value:g}. "
                "Its SHAP value indicates how this feature contributed "
                "to the model output."
            ),
        }

    if feature == "viral_load_change":
        value = float(row["viral_load_change"])

        return {
            "current_value": value,
            "patient_explanation": (
                f"The supplied viral-load-change feature is {value:g}. "
                "Its SHAP value indicates how this feature contributed "
                "to the model output."
            ),
        }

    return {
        "patient_explanation": (
            "This feature contributed to the model output as indicated "
            "by its SHAP value. Its contribution is not proof of causation."
        ),
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok" if pipeline is not None else "error",
        "service": "treatment-failure-prediction",
        "model": "Gradient Boosting",
        "model_loaded": pipeline is not None,
        "model_error": model_load_error,
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):
    model_pipeline = get_pipeline()

    try:
        input_df = request_to_dataframe(request)
        prediction = model_pipeline.predict(input_df)[0]

        result = {
            "prediction": int(prediction),
            "model": "Gradient Boosting",
        }

        if hasattr(model_pipeline, "predict_proba"):
            probabilities = model_pipeline.predict_proba(input_df)[0]

            result["class_probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(
                    model_pipeline.classes_,
                    probabilities,
                )
            }

            predicted_index = list(model_pipeline.classes_).index(
                prediction
            )

            result["probability"] = float(
                probabilities[predicted_index]
            )

        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


# --------------------------------------------------
# SHAP explanation endpoint
# --------------------------------------------------

@app.post("/explanation")
def explanation(request: PredictionRequest):
    model_pipeline = get_pipeline()

    try:
        input_df = request_to_dataframe(request)
        prediction = model_pipeline.predict(input_df)[0]

        transformed, feature_names = get_transformed_features(
            model_pipeline,
            input_df,
        )

        classifier = model_pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)

        classes = list(model_pipeline.classes_)
        class_index = classes.index(prediction)

        # Support common SHAP output formats.
        if isinstance(shap_values, list):
            values_for_row = np.asarray(
                shap_values[class_index]
            )[0]

        else:
            values_array = np.asarray(shap_values)

            if values_array.ndim == 3:
                # Format: samples x features x classes.
                if values_array.shape[2] == len(classes):
                    values_for_row = values_array[
                        0, :, class_index
                    ]

                # Format: samples x classes x features.
                elif values_array.shape[1] == len(classes):
                    values_for_row = values_array[
                        0, class_index, :
                    ]

                else:
                    raise ValueError(
                        "Cannot identify the class dimension in "
                        f"SHAP output shape {values_array.shape}."
                    )

            elif values_array.ndim == 2:
                # Binary tree models can return a single output.
                values_for_row = values_array[0]

            else:
                raise ValueError(
                    f"Unexpected SHAP output shape: "
                    f"{values_array.shape}"
                )

        values_for_row = np.asarray(values_for_row).reshape(-1)

        if len(values_for_row) != len(feature_names):
            raise ValueError(
                "The number of SHAP values does not match "
                "the feature names."
            )

        contributions = []

        for index, feature_name in enumerate(feature_names):
            shap_value = float(values_for_row[index])

            feature_info = describe_shap_feature(feature_name)
            patient_info = explain_patient_feature(
                feature_name,
                input_df,
            )

            if shap_value > 0:
                direction = "positive"
            elif shap_value < 0:
                direction = "negative"
            else:
                direction = "neutral"

            contributions.append({
                "feature": feature_name,
                "feature_label": feature_info["feature_label"],
                "description": feature_info["description"],
                "shap_value": shap_value,
                "absolute_shap_value": abs(shap_value),
                "shap_direction": direction,
                "patient_context": patient_info,
            })

        contributions.sort(
            key=lambda item: item["absolute_shap_value"],
            reverse=True,
        )

        return {
            "prediction": int(prediction),
            "model": "Gradient Boosting",
            "explanation_method": "SHAP TreeExplainer",
            "top_features": contributions[:10],
            "all_features": contributions,
            "note": (
                "SHAP values describe contributions to the model "
                "output being explained. Positive and negative values "
                "indicate contribution direction, not necessarily higher "
                "or lower clinical treatment-failure risk. The feature "
                "descriptions summarize recorded data and do not establish "
                "causation or independently confirm clinical failure."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {exc}",
        ) from exc
