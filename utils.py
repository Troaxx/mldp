import pandas as pd
import joblib
import streamlit as st

def load_model_and_scaler():
    """
    Loads the saved Logistic Regression model and StandardScaler using joblib.
    """
    try:
        model = joblib.load('hospital_readmission_model.pkl')
        scaler = joblib.load('scaler.pkl')
            
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

def preprocess_input(data, scaler):
    """
    Preprocesses the input dictionary into a dataframe suitable for prediction.
    Includes feature engineering, encoding, and scaling.
    """
    df = pd.DataFrame([data])
    
    # --- Feature Engineering ---
    # Derived from eda.ipynb relationships
    df['total_medications'] = df['num_medications']
    df['total_procedures'] = df['num_procedures'] + df['num_lab_procedures']
    df['service_utilization'] = (df['number_outpatient'] + 
                                 df['number_emergency'] + 
                                 df['number_inpatient'])
    df['diagnosis_complexity'] = df['number_diagnoses']
    
    df['has_previous_admission'] = (df['number_inpatient'] > 0).astype(int)
    df['has_emergency_visit'] = (df['number_emergency'] > 0).astype(int)
    
    # --- Encoding ---
    # Age Mapping
    age_mapping = {
        '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
        '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
    }
    # Handle both string input from dropdown or direct integer input if passed
    if df['age'].dtype == 'O':
         df['age_encoded'] = df['age'].map(age_mapping)
    else:
         df['age_encoded'] = df['age'] # Assume already encoded if not string

    # Gender Mapping
    gender_mapping = {'Male': 1, 'Female': 0, 'Unknown/Invalid': 0.5}
    df['gender_encoded'] = df['gender'].map(gender_mapping)
    
    # --- Feature Selection (Order matters for scaler) ---
    feature_columns = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses',
        'total_medications', 'total_procedures', 'service_utilization',
        'diagnosis_complexity', 'has_previous_admission', 'has_emergency_visit',
        'age_encoded', 'gender_encoded'
    ]
    
    # specific handling if mapping failed (e.g. unexpected input)
    if df['age_encoded'].isnull().any():
         # Fallback or error - for now fill with median-like value or 0
         df['age_encoded'] = df['age_encoded'].fillna(0) 
    if df['gender_encoded'].isnull().any():
         df['gender_encoded'] = df['gender_encoded'].fillna(0.5)

    X = df[feature_columns]
    
    # --- Scaling ---
    X_scaled = scaler.transform(X)
    
    return X_scaled
