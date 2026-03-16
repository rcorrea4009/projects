import streamlit as st
import joblib
import numpy as np
import os
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern, hog

# 1. Page Configuration
st.set_page_config(page_title="Rice Leaf Disease Detector", page_icon="🌾")

# Styling
st.markdown("""
    <style>
    .stButton>button { 
        background-color: #2e7d32 !important; color: white !important; 
        border-radius: 8px; font-weight: bold; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Model Loader
@st.cache_resource
def load_rice_model():
    # Make sure this matches your .pkl filename exactly
    model_path = os.path.join(os.path.dirname(__file__), 'rice_disease_traditional_model.pkl')
    return joblib.load(model_path)

def load_lable_model():
    # Make sure this matches your .pkl filename exactly
    model_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')
    return joblib.load(model_path)

def load_scaler_model():
    # Make sure this matches your .pkl filename exactly
    model_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
    return joblib.load(model_path)

scaler = load_scaler_model()
le = load_lable_model()

try:
    model = load_rice_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. EXACT FEATURE EXTRACTION FUNCTIONS (From your Notebook)
def extract_all_features(image_np):
    # Resize to 224x224 as per your notebook code
    img = cv2.resize(image_np, (224, 224))
    features = []

    # A. Color Features (30 features)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    for channel_set in [rgb, hsv, lab]:
        for i in range(3):
            ch = channel_set[:, :, i]
            if channel_set is rgb:
                features.extend([np.mean(ch), np.std(ch), np.median(ch), 
                                 np.percentile(ch, 25), np.percentile(ch, 75)])
            elif channel_set is hsv:
                features.extend([np.mean(ch), np.std(ch), np.median(ch)])
            else: # LAB
                features.extend([np.mean(ch), np.std(ch)])

    # B. Texture Features (14 features)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    radius, n_points = 3, 24
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
    features.extend(lbp_hist[:10]) 
    
    hog_feats = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    features.extend([np.mean(hog_feats), np.std(hog_feats), np.max(hog_feats), np.min(hog_feats)])

    # C. Edge Features (12 features)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    canny = cv2.Canny(gray, 50, 150)
    
    for edges in [sobelx, sobely, laplacian, canny]:
        features.extend([np.mean(np.abs(edges)), np.std(edges), np.sum(np.abs(edges) > 0) / edges.size])

    return np.array(features).reshape(1, -1)

# 4. Header & Upload
st.title("🌾 Rice Leaf Disease Prediction")
uploaded_file = st.file_uploader("Upload leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


    
    if st.button("Analyze Leaf Health"):
        with st.spinner('Calculating 56 features...'):
            try:
                # Get features
                final_features = extract_all_features(img_cv)
                features_scaled = scaler.transform(final_features)
                # 1. Get the numerical prediction
                prediction_idx = model.predict(features_scaled)[0]
                
                # 2. Map the number to the real name
                # This list MUST be in the same order as your training folders
                class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
                
                # Safety check: if prediction is already a string, use it; if it's a number, map it
                if isinstance(prediction_idx, (int, np.integer)):
                    result_text = class_names[prediction_idx]
                else:
                    result_text = le.inverse_transform([prediction_idx])[0]

                st.divider()
                
                # 3. Display the clean result
                if "Healthy" in str(result_text):
                    st.success(f"### Result: {result_text}")
                else:
                    st.error(f"### Detected: {result_text}")
                
                # 4. Confidence Score
                if hasattr(model, "predict_proba"):
                    prob = np.max(model.predict_proba(final_features))
                    st.write(f"**Confidence:** {prob:.1%}")
                    st.progress(prob)

            except Exception as e:
                st.error(f"Error during analysis: {e}")