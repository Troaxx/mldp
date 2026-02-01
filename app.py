import streamlit as st
import pandas as pd
from utils import load_model_and_scaler, preprocess_input

# Page Configuration
st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model and Scaler (Cached)
@st.cache_resource
def get_model_and_scaler():
    return load_model_and_scaler()

model, scaler = get_model_and_scaler()

# Title and Introduction
st.title("🏥 Hospital Readmission Prediction")
st.markdown("""
    This application predicts the likelihood of a patient being readmitted to the hospital within 30 days.
    Please enter the patient's clinical and demographic details below.
""")

st.write("---")

# Input Form
with st.form("prediction_form"):
    st.subheader("Patient Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        age_ranges = [
            '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
            '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
        ]
        age = st.selectbox("Age Group", age_ranges, index=6)
        
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male", "Unknown/Invalid"])

    st.subheader("Clinical Metrics")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=365, value=3)
        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, value=10)
        num_procedures = st.number_input("Number of Procedures", min_value=0, value=0)
        
    with col4:
        num_medications = st.number_input("Number of Medications", min_value=0, value=5)
        number_diagnoses = st.number_input("Number of Diagnoses", min_value=0, value=1)
        
    with col5:
        st.markdown("**History of Visits (Prior Year)**")
        number_outpatient = st.number_input("Number of Outpatient Visits", min_value=0, value=0)
        number_emergency = st.number_input("Number of Emergency Visits", min_value=0, value=0)
        number_inpatient = st.number_input("Number of Inpatient Visits", min_value=0, value=0)

    st.write("---")
    submitted = st.form_submit_button("Predict Readmission Risk", type="primary")

# Prediction Logic
if submitted:
    if model is not None and scaler is not None:
        # Prepare input data dictionary (keys match utils.py expectations)
        input_data = {
            'age': age,
            'gender': gender,
            'time_in_hospital': time_in_hospital,
            'num_lab_procedures': num_lab_procedures,
            'num_procedures': num_procedures,
            'num_medications': num_medications,
            'number_outpatient': number_outpatient,
            'number_emergency': number_emergency,
            'number_inpatient': number_inpatient,
            'number_diagnoses': number_diagnoses
        }
        
        try:
            # Preprocess
            X_processed = preprocess_input(input_data, scaler)
            
            # Predict
            prediction = model.predict(X_processed)[0]
            probability = model.predict_proba(X_processed)[0][1]
            
            # Display Result
            st.header("Prediction Result")
            if prediction == 1:
                st.error(f"**High Risk**: The model predicts this patient **WILL** be readmitted within 30 days.")
                st.write(f"**Probability:** {probability:.2%}")
            else:
                st.success(f"**Low Risk**: The model predicts this patient will **NOT** be readmitted within 30 days.")
                st.write(f"**Probability:** {probability:.2%}")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Model or Scaler not loaded. Please check if the .pkl files exist.")

# Footer/Disclaimer
st.markdown("---")
st.caption("Note: This tool is for educational/demonstration purposes only and should not be used as a sole medical diagnostic tool.")
