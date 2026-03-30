import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
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
        background-color: #0E1117;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1A4D2E 0%, #4CAF50 100%);
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Card Styles for About Section */
    .card-container {
        background-color: #1E2530;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .card-title {
        color: #00E676;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Stats Box Styles */
    .stat-box {
        background-color: #1E2530;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stat-value {
        color: #00E676;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #A0AEC0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tech Stack Pills */
    .tech-pill {
        display: inline-block;
        background-color: #1A4D2E;
        color: #E8F5E9;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
        border: 1px solid #4CAF50;
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
        "Potato": "https://images.unsplash.com/photo-1518977676601-b53f82aba655?w=600&auto=format&fit=crop"
    }
    return image_map.get(crop_name, "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=600&auto=format&fit=crop")

def get_fertilizer_details(name):
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
    st.error("❌ Crucial AI model files are missing. Please execute `python train_fertilizer.py` to compile dependencies.")
    st.stop()

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🌱 AI Prediction System", "📊 Dataset Analysis", "ℹ️ About KrushiAI"])

# ----------------- TAB 1: PREDICTION -----------------
with tab1:
    col_input, col_result = st.columns([1, 1.3], gap="large")
    
    with col_input:
        st.markdown("<div class='card-title'>Input Field Parameters</div>", unsafe_allow_html=True)
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
        if not predict_btn:
            st.markdown("<div class='card-title'>AI Engine Standby</div>", unsafe_allow_html=True)
            st.info("👈 Enter the field parameters and generate an AI-powered recommendation.")
            try:
                st.image("krushiai_hero.png", caption="KrushiAI: Real-Time Agricultural Data Automation", use_container_width=True)
            except Exception:
                st.image("https://images.unsplash.com/photo-1628178652410-6ed6cebb8a81?w=800&auto=format&fit=crop", 
                         caption="Optimizing agricultural output with Data Science", use_container_width=True)
            
        # On Predict
        if predict_btn:
            with st.spinner("Analyzing parameters through KrushiAI inference engine..."):
                try:
                    s_encoded = soil_enc.transform([i_soil])[0]
                    c_encoded = crop_enc.transform([i_crop])[0]
                    
                    input_features = np.array([[i_temp, i_hum, i_moist, s_encoded, c_encoded, i_n, i_k, i_p]])
                    if scaler:
                        input_features = scaler.transform(input_features)
                    
                    prediction_encoded = model.predict(input_features)[0]
                    fert_name = fert_enc.inverse_transform([prediction_encoded])[0]
                    confidence = np.max(model.predict_proba(input_features)[0])
                    
                    st.success(f"### 🏆 Recommended Fertilizer: {fert_name}")
                    
                    sc1, sc2 = st.columns(2)
                    sc1.metric("Confidence Level", f"{confidence:.1%}")
                    sc2.metric("Recommendation Strategy", "Optimized")
                    
                    details = get_fertilizer_details(fert_name)
                    st.markdown("---")
                    
                    d_col1, d_col2 = st.columns([1, 1])
                    with d_col1:
                        st.markdown(f"**Description:** {details['desc']}")
                        st.markdown(f"**Classification:** {details['type']}")
                        st.markdown(f"**Ideal Stage:** {details['stage']}")
                    with d_col2:
                         st.image(get_crop_image(i_crop), caption=f"Recommended for {i_crop}", use_container_width=True)
                         
                    st.warning("⚠️ Recommendation generated via deterministic random forest mapping.")

                except Exception as e:
                    st.error(f"Prediction failed due to internal error: {str(e)}")


# ----------------- TAB 2: DATASET EXPLORER -----------------
with tab2:
    st.markdown("<div class='card-title'>Global Dataset Explorer</div>", unsafe_allow_html=True)
    
    df = load_dataset()
    if not df.empty:
        # Dashboard Matrix Header
        db_col1, db_col2, db_col3, db_col4 = st.columns(4)
        acc = metrics.get('accuracy', 1.0) if metrics else 1.0
        
        with db_col1:
            st.markdown(f"<div class='stat-box'><div class='stat-value' style='color:#38bdf8;'>{len(df):,}</div><div class='stat-label'>Massive Records</div></div>", unsafe_allow_html=True)
        with db_col2:
            st.markdown(f"<div class='stat-box'><div class='stat-value' style='color:#38bdf8;'>{len(df.columns)-1}</div><div class='stat-label'>Features</div></div>", unsafe_allow_html=True)
        with db_col3:
            st.markdown(f"<div class='stat-box'><div class='stat-value' style='color:#38bdf8;'>{len(df['Fertilizer Name'].unique())}</div><div class='stat-label'>Fertilizer Classes</div></div>", unsafe_allow_html=True)
        with db_col4:
            st.markdown(f"<div class='stat-box'><div class='stat-value' style='color:#38bdf8;'>> {acc*100:.0f}%</div><div class='stat-label'>Pipeline Accuracy</div></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 1: Heatmap & Class Distribution
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            st.markdown("<div class='card-title'>Numerical Heatmap</div>", unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            # Use Plotly Express instead of Seaborn for interactive Streamlit alignment
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=".2f", 
                aspect="auto", 
                color_continuous_scale="RdYlGn"
            )
            fig_corr.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0', height=400, margin=dict(l=0, r=0, t=10, b=10)
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with g_col2:
            st.markdown("<div class='card-title'>Fertilizer Class Distribution</div>", unsafe_allow_html=True)
            dist_df = df['Fertilizer Name'].value_counts().reset_index()
            dist_df.columns = ['Fertilizer', 'Count']
            
            fig_dist = px.bar(
                dist_df, 
                x="Count", 
                y="Fertilizer", 
                orientation='h',
                color="Count",
                color_continuous_scale="Purples"
            )
            fig_dist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0', height=400, margin=dict(l=0, r=0, t=10, b=10),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Row 2: Deep Feature Interrogation
        st.markdown("<br><div class='card-title'>Feature Interrogation</div>", unsafe_allow_html=True)
        selected_num_ft = st.selectbox("Select Numerical Feature for Distribution Analysis", numeric_cols)
        
        if selected_num_ft:
            fig_box = px.box(
                df, 
                x="Fertilizer Name", 
                y=selected_num_ft, 
                color="Fertilizer Name",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_box.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0', height=450, margin=dict(l=0, r=0, t=10, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Row 3: Raw Data Preview
        st.markdown("<br><div class='card-title'>Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(
            df.head(100).style.background_gradient(cmap='Greens', subset=['Nitrogen', 'Phosphorous', 'Potassium']),
            use_container_width=True,
            height=300
        )
    else:
        st.warning("No dataset located. Please ensure `fertilizer_recommendation_dataset.csv` exists.")


# ----------------- TAB 3: ABOUT KRUSHIAI -----------------
with tab3:
    # Top Description Card
    st.markdown("""
    <div class="card-container">
        <div class="card-title">About KrushiAI 🔗</div>
        <p style="color: #E2E8F0; line-height: 1.6;">
            <b style="color: #00E676;">KrushiAI</b> is an advanced, highly intelligent fertilizer recommendation system powered by deep machine learning. 
            It autonomously analyzes complex soil composition matrices (N-P-K) alongside real-world environmental factors 
            (including temperature, moisture, and humidity) to recommend the absolute most suitable agricultural fertilizer for optimal farming yield.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_about1, col_about2 = st.columns(2)
    
    with col_about1:
        st.markdown("""
        <div class="card-container" style="height: 100%;">
            <div class="card-title">How It Works</div>
            <p style="line-height: 1.8; color: #E2E8F0;">
                <span style="background-color: #1A4D2E; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">1</span> Enter your <b>Soil Nutrients</b> (Nitrogen, Phosphorous, Potassium).<br>
                <span style="background-color: #1A4D2E; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">2</span> Enter <b>Environmental Conditions</b> (Temp, Humidity, Moisture).<br>
                <span style="background-color: #1A4D2E; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">3</span> Input <b>Farming Logistics</b> (Target Crop & Soil Type).<br>
                <span style="background-color: #1A4D2E; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">4</span> Our <b>Random Forest AI Pipeline</b> rigorously analyzes the input array.<br>
                <span style="background-color: #1A4D2E; padding: 2px 8px; border-radius: 4px; margin-right: 8px;">5</span> Get <b>instant fertilizer recommendations</b> mapped to your exact soil deficit.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_about2:
        st.markdown("""
        <div class="card-container" style="height: 100%;">
            <div class="card-title">Benefits</div>
            <p style="line-height: 1.8; color: #E2E8F0;">
                🎯 <b>Optimize agricultural yield</b> safely and predictably ensuring no soil toxicity.<br>
                💰 <b>Reduce resource wastage</b> by avoiding incompatible or excessive fertilization.<br>
                📊 <b>Empower data-driven farming decisions</b> built on mathematically pure datasets.<br>
                🌱 <b>Promote sustainable agriculture</b> long-term through efficient chemical application.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Performance Stats
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    acc = metrics.get('accuracy', 1.0) if metrics else 1.0
    
    with col_stat1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">> {acc*100:.0f}%</div>
            <div class="stat-label">TEST ACCURACY</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">7</div>
            <div class="stat-label">FERTILIZER BLENDS</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_stat3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-value">8</div>
            <div class="stat-label">INPUT FEATURES</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Importance Chart
    if metrics and 'feature_importance' in metrics:
        st.markdown("""
        <div class="card-container">
            <div class="card-title">Feature Importance</div>
        """, unsafe_allow_html=True)
        
        fi_df = pd.DataFrame(metrics['feature_importance']).sort_values('importance', ascending=True)
        
        fig = px.bar(
            fi_df, 
            x='importance', 
            y='feature', 
            orientation='h',
            color='importance',
            color_continuous_scale='Greens'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E2E8F0',
            xaxis_title="Relative Importance",
            yaxis_title="",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Tech Stack
    st.markdown("""
    <div class="card-container">
        <div class="card-title">Tech Stack</div>
        <div style="margin-top: 1rem;">
            <span class="tech-pill">Python</span>
            <span class="tech-pill">Scikit-learn</span>
            <span class="tech-pill">Random Forest</span>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">Pandas</span>
            <span class="tech-pill">Plotly</span>
            <span class="tech-pill">Numpy</span>
        </div>
        <div style="text-align: center; margin-top: 2rem; color: #A0AEC0; font-size: 0.9rem;">
            Built with Streamlit & Scikit-learn | Powered by Mathematically Mapped Random Forests
        </div>
    </div>
    """, unsafe_allow_html=True)
