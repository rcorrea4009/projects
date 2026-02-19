import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

base_path = os.path.dirname(__file__)

# 1. Join and load that path with your file names
model_path = os.path.join(base_path, 'insurance_cost_model.pkl')
preprocessor_path = os.path.join(base_path, 'preprocessor.pkl')

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# UI Title and Description
st.title("🏥 Insurance Cost Predictor")
st.write("Enter the details below to estimate insurance charges.")

# 2. Create the input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5])
        
    with col2:
        sex = st.selectbox("Sex", options=['male', 'female'])
        smoker = st.selectbox("Smoker?", options=['yes', 'no'])
        region = st.selectbox("Region", options=['southwest', 'southeast', 'northwest', 'northeast'])
    
    submit = st.form_submit_button("Predict Insurance Cost")

# 3. Handle Prediction
if submit:
    # Create a DataFrame for the input
    input_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])
    
    # Preprocess the input data
    # Note: Ensure the preprocessor handles the data as it was during training
    processed_data = preprocessor.transform(input_df)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Display result
    st.success(f"### Estimated Insurance Cost: ${prediction[0]:,.2f}")
    
    # Contextual feedback based on BMI/Smoker status
    if smoker == 'yes':
        st.warning("Note: Smoking significantly increases insurance premiums.")
    if bmi > 30:
        st.info("Note: A BMI over 30 is classified as obese and may impact your rates.")