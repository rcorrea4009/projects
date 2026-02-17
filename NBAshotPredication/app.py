import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Kobe Bryant Shot Predictor", page_icon="🏀", layout="centered")

# Custom CSS for a professional sports dashboard look
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        background-color: #552583;
        color: #FDB927;
        font-weight: bold;
        border-radius: 5px;
        border: none;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource


def load_model():
    # Note: Ensure this file is in the same directory or provide the full path
    return joblib.load('kobe_shot_prediction_model.pkl')

base_path = os.path.dirname(__file__)
model_pipeline = load_model()

# Header
st.title("🏀 Kobe Bryant Shot Predictor")
st.markdown("Predict the likelihood of a shot being made based on game conditions and shot location.")
st.divider()

# Sidebar for Game Context
st.sidebar.header("Game Context")
playoffs = st.sidebar.selectbox("Is it a Playoff Game?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
period = st.sidebar.slider("Period", 1, 4, 1)
minutes_remaining = st.sidebar.slider("Minutes Remaining in Period", 0, 11, 5)
seconds_remaining = st.sidebar.slider("Seconds Remaining", 0, 59, 30)

# Main input section
st.subheader("Shot Details")
col1, col2 = st.columns(2)

with col1:
    loc_x = st.number_input("Location X (Horizontal distance from basket)", value=0)
    loc_y = st.number_input("Location Y (Vertical distance from basket)", value=0)
    shot_distance = st.number_input("Shot Distance (ft)", min_value=0, max_value=90, value=15)

with col2:
    combined_shot_type = st.selectbox("Shot Type", 
                                    ['Jump Shot', 'Dunk', 'Layup', 'Tip Shot', 'Hook Shot', 'Bank Shot'])
    shot_type = st.selectbox("Point Value", ['2PT Field Goal', '3PT Field Goal'])
    opponent = st.selectbox("Opponent", 
                           ['POR', 'UTA', 'VAN', 'LAC', 'HOU', 'SAS', 'DEN', 'SAC', 'CHI', 'GSW', 'MIN', 'IND', 'SEA', 'DAL', 'PHI', 'DET', 'TOR', 'PHX', 'ATL', 'MIL', 'NJN', 'ORL', 'CHH', 'BKN', 'WAS', 'OKC', 'MEM', 'CLE', 'NOH', 'CHA', 'BOS', 'NYK', 'NOK'])

# Predict button
if st.button("Predict Shot Outcome", use_container_width=True):
    # Prepare input data matching the training features identified in EDA
    input_data = pd.DataFrame([{
        'loc_x': loc_x,
        'loc_y': loc_y,
        'period': period,
        'playoffs': playoffs,
        'minutes_remaining': minutes_remaining,
        'seconds_remaining': seconds_remaining,
        'shot_distance': shot_distance,
        'combined_shot_type': combined_shot_type,
        'shot_type': shot_type,
        'opponent': opponent
    }])

    try:
        # Get prediction and probability
        prediction = model_pipeline.predict(input_data)[0]
        probability = model_pipeline.predict_proba(input_data)[0][1]

        st.divider()
        
        if prediction == 1:
            st.success(f"### 🔥 SWISH! Predicted: MADE")
        else:
            st.error(f"### 🧱 CLANK! Predicted: MISSED")
            
        st.write(f"**Confidence Score:** {probability:.1%}")
        
        # Progress bar for visual probability
        st.progress(probability)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Check if the feature names in input_data match exactly what the model was trained on.")

# Footer information
st.caption("This model was trained on Kobe Bryant's career shot data to analyze scoring patterns.")