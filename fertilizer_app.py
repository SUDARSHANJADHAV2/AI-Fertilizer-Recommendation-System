import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="KrushiAI - Smart Fertilizer Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CSS STYLING
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Button custom hover */
    .stButton > button {
        transition: all 0.3s ease;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Helper Functions
# ==============================================================================
@st.cache_data
def load_dataset():
    filepath = "fertilizer_recommendation_dataset.csv"
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()

@st.cache_resource
def load_model_components():
    try:
        model = pickle.load(open("Fertilizer_RF.pkl", "rb"))
        soil_encoder = pickle.load(open("soil_encoder.pkl", "rb"))
        crop_encoder = pickle.load(open("crop_encoder.pkl", "rb"))
        fertilizer_encoder = pickle.load(open("fertilizer_encoder.pkl", "rb"))
        
        try:
            scaler = pickle.load(open("feature_scaler.pkl", "rb"))
            model_metrics = pickle.load(open("model_metrics.pkl", "rb"))
        except:
            scaler, model_metrics = None, None
            
        return model, soil_encoder, crop_encoder, fertilizer_encoder, scaler, model_metrics
    except Exception as e:
        return None, None, None, None, None, None

def get_crop_image(crop_name):
    # Mapping real public URLs to crop types
    image_map = {
        "Wheat": "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=600&auto=format&fit=crop",
        "Maize": "https://images.unsplash.com/photo-1582281295992-ccb1ef6dfcb5?w=600&auto=format&fit=crop",
        "Cotton": "https://images.unsplash.com/photo-1596711585864-42f7c00eb13e?w=600&auto=format&fit=crop",
        "Sugarcane": "https://images.unsplash.com/photo-1592398555981-d1c95eceec1f?w=600&auto=format&fit=crop",
        "Tobacco": "https://images.unsplash.com/photo-1581451000806-05634063c1a2?w=600&auto=format&fit=crop",
        "Paddy": "https://images.unsplash.com/photo-1586208006456-11f496735a12?w=600&auto=format&fit=crop",
        "Barley": "https://images.unsplash.com/photo-1542385151-efd9000785a0?w=600&auto=format&fit=crop",
        "Coffee": "https://images.unsplash.com/photo-1497935586351-b67a49e012bf?w=600&auto=format&fit=crop",
        "Mango": "https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?w=600&auto=format&fit=crop",
        "Banana": "https://images.unsplash.com/photo-1528825871115-3581a5387919?w=600&auto=format&fit=crop",
        "Tomato": "https://images.unsplash.com/photo-1592841200221-a6898f307baa?w=600&auto=format&fit=crop",
        "Potato": "https://images.unsplash.com/photo-1518977676601-b53f82aba655?w=600&auto=format&fit=crop",
        "Apple": "https://images.unsplash.com/photo-1560806887-1e4cd0b6fac6?w=600&auto=format&fit=crop"
    }
    
    # default farm picture if exact match not found
    return image_map.get(crop_name, "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=600&auto=format&fit=crop")

def get_fertilizer_details(name):
    # Dictionary containing fertilizer properties
    data = {
        "Urea": {"desc": "High nitrogen (46-0-0) fertilizer.", "type": "Nitrogenous", "stage": "Top dressing"},
        "DAP": {"desc": "Di-ammonium Phosphate (18-46-0).", "type": "Phosphatic", "stage": "Basal application"},
        "14-35-14": {"desc": "Complex NPK blend.", "type": "Complex", "stage": "Pre-planting"},
        "28-28": {"desc": "NP blend.", "type": "Complex NP", "stage": "Vegetative"},
        "17-17-17": {"desc": "Balanced NPK.", "type": "Balanced", "stage": "All growth stages"},
        "20-20": {"desc": "NP blend suited for cereals.", "type": "Complex NP", "stage": "Sowing"},
        "10-26-26": {"desc": "PK heavy blend.", "type": "Complex PK", "stage": "Flowering / Fruit formation"}
    }
    return data.get(name, {"desc": "Custom formulation", "type": "Variable", "stage": "As needed"})

# ==============================================================================
# APP STRUCTURE
# ==============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🌾 KrushiAI Dashboard</h1>
    <p class="main-subtitle">AI-Driven Fertilizer Recommendation & Crop Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Load context
model, soil_enc, crop_enc, fert_enc, scaler, metrics = load_model_components()

if model is None:
    st.error("❌ Crucial AI model files are missing. Have you executed `python train_fertilizer.py`?")
    st.stop()

# Build layout
tab1, tab2, tab3 = st.tabs(["🔮 AI Prediction System", "📊 Database Explorer", "📚 Guide & Instructions"])

# ----------------- TAB 1: PREDICTION -----------------
with tab1:
    col_input, col_result = st.columns([1, 1.3], gap="large")
    
    with col_input:
        st.markdown("### 🌱 Provide Field Parameters")
        st.info("Input real-time data from your soil tests and environment.")
        
        with st.expander("🌡️ Environmental Conditions", expanded=True):
            i_temp = st.slider("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0, step=0.5)
            i_hum = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            i_moist = st.slider("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
            
        with st.expander("🧪 Soil Nutrients (mg/kg)", expanded=True):
            i_n = st.number_input("Nitrogen (N) ", min_value=0.0, max_value=300.0, value=40.0, step=1.0)
            i_p = st.number_input("Phosphorus (P) ", min_value=0.0, max_value=300.0, value=20.0, step=1.0)
            i_k = st.number_input("Potassium (K) ", min_value=0.0, max_value=300.0, value=20.0, step=1.0)
            
        with st.expander("🌾 Crop & Soil Context", expanded=True):
            i_soil = st.selectbox("Soil Type", options=soil_enc.classes_)
            i_crop = st.selectbox("Intended Crop Target", options=crop_enc.classes_)

        predict_btn = st.button("Generate Recommendation", use_container_width=True, type="primary")

    with col_result:
        # Before Predict - show metrics and image template
        if not predict_btn:
            st.markdown("### 📈 Model Health Check")
            if metrics:
                m1, m2 = st.columns(2)
                m1.metric("Validation Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                m2.metric("Mean CV Score", f"{metrics.get('cv_mean', 0):.2%}")
                
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info("👈 Enter the field parameters and generate an AI-powered recommendation.")
            st.image("https://images.unsplash.com/photo-1628178652410-6ed6cebb8a81?w=800&auto=format&fit=crop", 
                     caption="Optimizing agricultural output with Data Science", use_container_width=True)
            
        # On Predict
        if predict_btn:
            with st.spinner("Analyzing parameters through KrushiAI inference engine..."):
                try:
                    # Encode categorical inputs
                    s_encoded = soil_enc.transform([i_soil])[0]
                    c_encoded = crop_enc.transform([i_crop])[0]
                    
                    # Form feature matrix
                    input_features = np.array([[i_temp, i_hum, i_moist, s_encoded, c_encoded, i_n, i_k, i_p]])
                    
                    if scaler:
                        input_features = scaler.transform(input_features)
                    
                    # Inference
                    prediction_encoded = model.predict(input_features)[0]
                    fert_name = fert_enc.inverse_transform([prediction_encoded])[0]
                    confidence = np.max(model.predict_proba(input_features)[0])
                    
                    # Output Render
                    st.success(f"### 🏆 Recommended Fertilizer: {fert_name}")
                    
                    sc1, sc2 = st.columns(2)
                    sc1.metric("Confidence Level", f"{confidence:.1%}")
                    
                    # Fertilizer specific info
                    details = get_fertilizer_details(fert_name)
                    st.markdown("---")
                    
                    d_col1, d_col2 = st.columns([1, 1])
                    with d_col1:
                        st.markdown(f"**Description:** {details['desc']}")
                        st.markdown(f"**Classification:** {details['type']}")
                        st.markdown(f"**Ideal Stage:** {details['stage']}")
                    with d_col2:
                         st.image(get_crop_image(i_crop), caption=f"Recommended for {i_crop}", use_container_width=True)
                         
                    st.warning("⚠️ Always perform localized soil validation testing periodically to tune actual application volumes. This is an algorithmic recommendation based on normalized datasets.")

                except Exception as e:
                    st.error(f"Prediction failed due to internal formatting error: {str(e)}")


# ----------------- TAB 2: DATASET EXPLORER -----------------
with tab2:
    st.markdown("### 📊 Database Explorer")
    st.write("Browse and analyze the historical agronomy data used to train the KrushiAI model.")
    
    df = load_dataset()
    if not df.empty:
        st.dataframe(
            df.style.background_gradient(cmap='Greens', subset=['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']),
            use_container_width=True,
            height=500
        )
        st.caption(f"Total verified data records: {len(df)}")
    else:
        st.warning("No dataset located. Please ensure `fertilizer_recommendation_dataset.csv` exists.")


# ----------------- TAB 3: GUIDE & INSTRUCTIONS -----------------
with tab3:
    st.markdown("### 📚 Agronomy Intelligence Guide")
    st.markdown("""
    **Core Philosophy:**
    Fertilizers are not universal. Balancing N-P-K (Nitrogen, Phosphorus, Potassium) directly impacts yield, resistance, and soil longevity.
    
    * **Nitrogen (N):** Essential for vegetative, leafy growth. High N fertilizers like Urea are critical during initial shoot development.
    * **Phosphorus (P):** Crucial for deep robust root systems and high-quality flower/seed formation. Required early in crop lifecycle (e.g. DAP).
    * **Potassium (K):** Strengthens stems, boosts water regulation mechanisms, and fortifies crops against disease and harsh conditions.
    
    **Using the System:**
    1. Determine parameters locally via modern soil tests (or pH/N-P-K probes).
    2. Supply the accurate moisture, humidity, and atmospheric temperature.
    3. Ensure intended crop type aligns with regional viability.
    4. KrushiAI executes Random Forest inference mapping localized features to the optimal industrial mixture.
    """)
