
import sqlite3
from datetime import date

import pandas as pd
import requests
import streamlit as st

from database.db import (
    initialize_database,
    add_patient,
    add_visit,
    add_prediction,
    get_patient_history,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="HIV Treatment Outcome Research",
    page_icon="🧬",
    layout="wide",
)

initialize_database()


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("HIV Treatment Outcome Prediction")

st.caption("Research prototype | Longitudinal clinical data")

st.warning(
    "This application is for research and educational purposes only. "
    "Predictions are model estimates, not clinical diagnoses "
    "or treatment recommendations."
)


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "New Assessment", "Patient History"],
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("Research Dashboard")

    st.write(
        "Welcome to the HIV Treatment Outcome Prediction Research Prototype."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Application Status", "Online")
    col2.metric("Model", "Gradient Boosting")
    col3.metric("Database", "SQLite")

    st.divider()

    st.subheader("Project Overview")

    st.markdown("""
    This application provides a research interface for:

    - Longitudinal clinical data assessment
    - Machine learning-based outcome prediction
    - SHAP-based model explanations
    - Patient visit history management
    """)

    st.info(
        "Start by selecting New Assessment from the sidebar. "
        "Use synthetic data for testing."
    )


# --------------------------------------------------
# NEW ASSESSMENT
# --------------------------------------------------

elif page == "New Assessment":

    st.header("New Patient Assessment")

    with st.form("assessment_form"):

        patient_id = st.text_input(
            "Pseudonymous Patient ID",
            placeholder="e.g. DEMO001",
        )

        visit_date = st.date_input(
            "Visit Date",
            value=date.today(),
        )

        st.subheader("Current Visit")

        col1, col2 = st.columns(2)

        with col1:

            cd4 = st.number_input(
                "Current CD4",
                min_value=0.0,
                value=500.0,
            )

            viral_load = st.number_input(
                "Current Viral Load",
                min_value=0.0,
                value=1000.0,
            )

            year = st.number_input(
                "Year",
                min_value=1900,
                max_value=2100,
                value=date.today().year,
                step=1,
            )

            sex = st.selectbox(
                "Sex (as encoded in the model dataset)",
                ["Male", "Female"],
            )

            visit_number = st.number_input(
                "Visit Number (VISIT feature)",
                min_value=1,
                value=180,
                step=1,
            )

        with col2:

            cd4_previous = st.number_input(
                "Previous CD4",
                min_value=0.0,
                value=520.0,
            )

            viral_load_previous = st.number_input(
                "Previous Viral Load",
                min_value=0.0,
                value=800.0,
            )

            years_since_previous_visit = st.number_input(
                "Years Since Previous Visit",
                min_value=0.0,
                value=1.0,
            )

        # Calculate changes automatically

        cd4_change = cd4 - cd4_previous

        viral_load_change = viral_load - viral_load_previous

        st.subheader("Calculated Changes")

        col1, col2 = st.columns(2)

        col1.metric("CD4 Change", cd4_change)

        col2.metric("Viral Load Change", viral_load_change)

        submitted = st.form_submit_button(
            "Generate Research Prediction",
            type="primary",
        )

    # --------------------------------------------------
    # PROCESS ASSESSMENT
    # --------------------------------------------------

    if submitted:

        if not patient_id.strip():

            st.error("Please enter a pseudonymous patient ID.")

        else:

            clinical_data = {
                "CD4": cd4,
                "viral_load": viral_load,
                "year": int(year),
                "CD4_previous": cd4_previous,
                "viral_load_previous": viral_load_previous,
                "CD4_change": cd4_change,
                "viral_load_change": viral_load_change,
                "years_since_previous_visit": years_since_previous_visit,
                "sex": sex,
                "VISIT": int(visit_number),
            }

            try:

                with st.spinner("Generating research prediction..."):

                    response = requests.post(
                        f"{API_URL}/predict",
                        json=clinical_data,
                        timeout=30,
                    )

                if response.status_code != 200:

                    st.error(
                        f"Prediction API error: {response.text}"
                    )

                else:

                    result = response.json()

                    # Save patient if not already present

                    try:

                        add_patient(patient_id.strip())

                    except sqlite3.IntegrityError:

                        # Existing patient ID is allowed
                        pass

                    # Save visit

                    visit_id = add_visit(
                        patient_id.strip(),
                        visit_date.isoformat(),
                        clinical_data,
                    )

                    # Save prediction

                    add_prediction(
                        visit_id,
                        result["prediction"],
                        result["probability"],
                        result["class_probabilities"],
                        result.get("model", "Gradient Boosting"),
                    )

                    st.success(
                        "Prediction received and saved."
                    )

                    # ----------------------------------
                    # PREDICTION RESULTS
                    # ----------------------------------

                    st.subheader("Prediction Result")

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Predicted Class",
                        result["prediction"],
                    )

                    col2.metric(
                        "Predicted Class Probability",
                        f'{result["probability"]:.2%}',
                    )

                    st.subheader("Class Probabilities")

                    probability_data = pd.DataFrame(
                        {
                            "Class": list(
                                result["class_probabilities"].keys()
                            ),
                            "Probability": list(
                                result["class_probabilities"].values()
                            ),
                        }
                    )

                    st.bar_chart(
                        probability_data.set_index("Class")
                    )

                    st.caption(
                        "This probability refers to the predicted class. "
                        "It is not a measure of clinical certainty."
                    )

                    st.info(
                        "This result is a research model estimate. "
                        "It must not be used independently for clinical decisions."
                    )

                    # ----------------------------------
                    # SHAP EXPLANATION
                    # ----------------------------------

                    st.subheader("Model Explanation (SHAP)")

                    st.caption(
                        "SHAP values explain contributions to the model's "
                        "raw score for class 1, not direct changes in "
                        "predicted probability. They do not establish causation."
                    )

                    try:

                        with st.spinner(
                            "Generating model explanation..."
                        ):

                            explanation_response = requests.post(
                                f"{API_URL}/explanation",
                                json=clinical_data,
                                timeout=60,
                            )

                        if explanation_response.status_code == 200:

                            explanation = explanation_response.json()

                            top_features = explanation["top_features"]

                            if top_features:

                                shap_df = pd.DataFrame(top_features)

                                # Sort by absolute SHAP contribution

                                shap_df = shap_df.sort_values(
                                    by="absolute_shap_value",
                                    ascending=True,
                                )

                                st.subheader(
                                    "Top Features Influencing the Model"
                                )

                                st.bar_chart(
                                    shap_df.set_index("description")[
                                        "absolute_shap_value"
                                    ],
                                    horizontal=True,
                                )

                                st.subheader(
                                    "SHAP Contribution Details"
                                )

                                st.dataframe(
                                    shap_df[
                                        [
                                            "description",
                                            "shap_value",
                                            "direction",
                                        ]
                                    ],
                                    use_container_width=True,
                                    hide_index=True,
                                )

                            else:

                                st.info(
                                    "No SHAP feature contributions were returned."
                                )

                        else:

                            st.warning(
                                "Prediction was saved, but the explanation "
                                "could not be generated."
                            )

                    except requests.exceptions.RequestException as error:

                        st.warning(
                            f"SHAP explanation request failed: {error}"
                        )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI. "
                    "Start the backend before submitting an assessment."
                )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"API request failed: {error}"
                )

            except Exception as error:

                st.error(
                    f"An unexpected error occurred: {error}"
                )


# --------------------------------------------------
# PATIENT HISTORY
# --------------------------------------------------

elif page == "Patient History":

    st.header("Patient History")

    patient_id = st.text_input(
        "Enter Pseudonymous Patient ID",
        placeholder="e.g. DEMO001",
    )

    if st.button("Retrieve History"):

        if not patient_id.strip():

            st.warning(
                "Please enter a patient ID."
            )

        else:

            try:

                history = get_patient_history(
                    patient_id.strip()
                )

                if not history:

                    st.info(
                        "No patient history found."
                    )

                else:

                    st.success(
                        f"Found {len(history)} visit record(s)."
                    )

                    history_df = pd.DataFrame(history)

                    st.subheader("Visit Records")

                    st.dataframe(
                        history_df,
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as error:

                st.error(
                    f"Unable to retrieve patient history: {error}"
                )
