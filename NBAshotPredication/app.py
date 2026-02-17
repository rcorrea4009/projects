import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Kobe Shot Predictor", page_icon="🏀", layout="centered")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'kobe_shot_prediction_model.pkl')
    return joblib.load(model_path)

try:
    model_pipeline = load_model()
except Exception as e:
    st.error("Model file not found. Please ensure 'kobe_shot_prediction_model.pkl' is in the project folder.")
    st.stop()

st.title("🏀 Kobe Bryant Shot Predictor")
st.markdown("Enter shot details to calculate the probability of a 'Swish'.")

with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        loc_x = st.number_input("X Coordinate", value=0)
        loc_y = st.number_input("Y Coordinate", value=0)
        shot_dist = st.slider("Shot Distance (ft)", 0, 90, 15)
        action_type = st.selectbox("Action Type", ['Jump Shot', 'Layup', 'Driving Layup', 'Dunk', 'Slam Dunk'])
    with c2:
        shot_cat = st.selectbox("Shot Category", ['Jump Shot', 'Dunk', 'Layup', 'Tip Shot', 'Hook Shot', 'Bank Shot'])
        pt_value = st.radio("Point Value", ['2PT Field Goal', '3PT Field Goal'], horizontal=True)
        opponent = st.selectbox("Opponent", sorted(['POR', 'UTA', 'HOU', 'SAS', 'SAC', 'PHX', 'GSW', 'BOS', 'NYK']))
        season = st.selectbox("Season", ['2015-16', '2012-13', '2009-10', '2005-06', '2000-01'])

with st.expander("🕒 Game Situation"):
    playoffs = st.toggle("Playoff Game", value=False)
    period = st.number_input("Period", 1, 4, 1)
    min_rem = st.number_input("Min Remaining", 0, 11, 5)
    sec_rem = st.number_input("Sec Remaining", 0, 59, 30)

if st.button("Analyze Shot Probability", use_container_width=True):
    # 1. BASIC INPUTS
    now = datetime.now()
    total_sec = (min_rem * 60) + sec_rem
    data = {
        # Raw Inputs
        'loc_x': loc_x, 'loc_y': loc_y, 'period': period,
        'playoffs': int(playoffs), 'minutes_remaining': min_rem,
        'seconds_remaining': sec_rem, 'shot_distance': shot_dist,
        'combined_shot_type': shot_cat, 'shot_type': pt_value,
        'opponent': opponent, 'action_type': action_type,
        'season': season,
        
        # Temporal Features (Model expects integers for dates)
        'game_year': now.year, 'game_month': now.month, 'game_day': now.day,
        'day_of_week': now.weekday(), # 0-6 integer
        
        # Time-based Features
        'total_seconds': total_sec,
        'is_last_minute': 1 if total_sec <= 60 else 0,
        'is_last_2_minutes': 1 if total_sec <= 120 else 0,
        'is_first_quarter': 1 if period == 1 else 0,
        'is_fourth_quarter': 1 if period == 4 else 0,
        'is_overtime': 1 if period > 4 else 0,
        
        # Location/Math Features
        'abs_loc_x': abs(loc_x),
        'distance_from_center': np.sqrt(loc_x**2 + loc_y**2),
        'distance_x_period': shot_dist * period,
        'distance_x_playoffs': shot_dist * int(playoffs),
        'is_three_point': 1 if pt_value == '3PT Field Goal' else 0,
        
        # Zone Mapping (Simplified defaults)
        'shot_zone_basic': 'Mid-Range' if shot_dist > 8 else 'Restricted Area',
        'shot_zone_area': 'Center(C)',
        'shot_zone_range': '16-24 ft.' if shot_dist > 16 else 'Less Than 8 ft.'
    }
    
    # Boolean location flags
    data['is_restricted_area'] = 1 if shot_dist < 4 else 0
    data['is_in_the_paint'] = 1 if shot_dist < 15 else 0
    data['is_mid_range'] = 1 if (shot_dist >= 8 and shot_dist < 24) else 0
    data['is_three_point'] = 1 if pt_value == '3PT Field Goal' else 0

    input_df = pd.DataFrame([data])

    try:
        prob = model_pipeline.predict_proba(input_df)[0][1]
        st.metric("Probability of Scoring", f"{prob:.1%}")
        st.progress(prob)
    except Exception as e:
        st.error(f"Error: {e}")