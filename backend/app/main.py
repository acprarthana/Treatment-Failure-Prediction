from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting_treatment_proxy.joblib"

app = FastAPI(
    title="HIV Treatment Failure Prediction API",
    description="Prediction and SHAP explanation for the treatment-outcome proxy model.",
    version="1.0.0",
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


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run src/save_final_model.py first."
        )

    saved_object = joblib.load(MODEL_PATH)

    # Support either a saved dictionary or a directly saved sklearn pipeline.
    if isinstance(saved_object, dict):
        pipeline = saved_object.get("pipeline")
        if pipeline is None:
            raise ValueError(
                "The saved model dictionary does not contain a 'pipeline' entry."
            )
    else:
        pipeline = saved_object

    if not isinstance(pipeline, Pipeline):
        raise TypeError("The saved model is not an sklearn Pipeline.")

    if "model" not in pipeline.named_steps:
        raise ValueError("The pipeline does not contain a step named 'model'.")

    return pipeline


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


def request_to_dataframe(request: PredictionRequest) -> pd.DataFrame:
    values = request.model_dump()
    return pd.DataFrame(
        [[values[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )


def get_transformed_features(model_pipeline, input_df):
    """Apply preprocessing without running the final prediction step."""
    preprocessing = model_pipeline[:-1]
    transformed = preprocessing.transform(input_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed)

    try:
        feature_names = preprocessing.get_feature_names_out()
        feature_names = [str(name) for name in feature_names]
    except Exception:
        feature_names = [
            f"feature_{index}" for index in range(transformed.shape[1])
        ]

    if len(feature_names) != transformed.shape[1]:
        feature_names = [
            f"feature_{index}" for index in range(transformed.shape[1])
        ]

    return transformed, feature_names


@app.get("/health")
def health():
    return {
        "status": "ok" if pipeline is not None else "error",
        "service": "treatment-failure-prediction",
        "model": "Gradient Boosting",
        "model_loaded": pipeline is not None,
        "model_error": model_load_error,
    }


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
            result["probability"] = float(
                probabilities[list(model_pipeline.classes_).index(prediction)]
            )
            result["class_probabilities"] = {
                str(label): float(probability)
                for label, probability in zip(
                    model_pipeline.classes_, probabilities
                )
            }

        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post("/explanation")
def explanation(request: PredictionRequest):
    model_pipeline = get_pipeline()

    try:
        input_df = request_to_dataframe(request)
        prediction = model_pipeline.predict(input_df)[0]

        transformed, feature_names = get_transformed_features(
            model_pipeline, input_df
        )

        classifier = model_pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)

        # SHAP can return a list or an array, depending on its version
        # and the classifier's output format.
        if isinstance(shap_values, list):
            class_index = list(model_pipeline.classes_).index(prediction)
            values_for_row = np.asarray(
                shap_values[min(class_index, len(shap_values) - 1)]
            )[0]
        else:
            values_array = np.asarray(shap_values)

            if values_array.ndim == 3:
                class_index = list(model_pipeline.classes_).index(prediction)
                values_for_row = values_array[
                    0, :, min(class_index, values_array.shape[2] - 1)
                ]
            elif values_array.ndim == 2:
                values_for_row = values_array[0]
            else:
                raise ValueError(
                    f"Unexpected SHAP output shape: {values_array.shape}"
                )

        if len(values_for_row) != len(feature_names):
            raise ValueError(
                "The number of SHAP values does not match the feature names."
            )

        contributions = [
            {
                "feature": feature_names[index],
                "shap_value": float(values_for_row[index]),
                "absolute_shap_value": float(abs(values_for_row[index])),
            }
            for index in range(len(feature_names))
        ]

        contributions.sort(
            key=lambda item: item["absolute_shap_value"],
            reverse=True,
        )

        return {
            "prediction": int(prediction),
            "model": "Gradient Boosting",
            "explanation_method": "SHAP TreeExplainer",
            "top_features": contributions[:10],
            "note": (
                "SHAP values describe how features influence this model's "
                "output. They do not establish causation or confirm clinical "
                "treatment failure independently."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {exc}",
        ) from exc
