"""
app.py
------------------------------------------------------
STEP 3 of 3.
This is the web app (frontend). It loads the 3 trained models and
lets the user enter values manually to get predictions.

Run this file AFTER train_models.py:
    streamlit run app.py
------------------------------------------------------
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Smart Farmer Assistant", page_icon="🌾", layout="wide")

# ------------------------------------------------------
# Make sure the models exist before doing anything else
# ------------------------------------------------------
if not os.path.exists("models/crop_model.pkl"):
    st.error("Models not found! Please run these two commands in your terminal first:\n\n"
              "python generate_data.py\n\npython train_models.py")
    st.stop()

# Load all 3 models once (cached so it's fast)
@st.cache_resource
def load_models():
    crop_model = joblib.load("models/crop_model.pkl")
    yield_model = joblib.load("models/yield_model.pkl")
    weather_model = joblib.load("models/weather_model.pkl")
    weather_scaler = joblib.load("models/weather_scaler.pkl")
    return crop_model, yield_model, weather_model, weather_scaler

crop_model, yield_model, weather_model, weather_scaler = load_models()


# ========================================================
# STYLING — fonts, colors, hero banner, cards
# All of this is just CSS injected into the page. It does
# not change how the app works, only how it looks.
# ========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --forest: #1B4332;
    --forest-light: #2D6A4F;
    --wheat: #C98A32;
    --wheat-light: #E3B23C;
    --sky: #3E7C97;
    --cream: #FBF8F1;
    --card: #FFFFFF;
    --soil: #3A2E22;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: var(--cream);
}

h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    color: var(--forest) !important;
}

/* ---------- Default body text (fixes white-on-light text) ---------- */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label,
.stMarkdown, div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] p {
    color: var(--soil) !important;
}

/* ---------- Hero banner ---------- */
.hero {
    background: linear-gradient(120deg, var(--forest) 0%, var(--forest-light) 100%);
    padding: 2.6rem 2.5rem;
    border-radius: 18px;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 24px rgba(27, 67, 50, 0.18);
}
.stApp .hero h1 {
    color: #FFFFFF !important;
    font-size: 2.4rem;
    margin: 0 0 0.4rem 0;
}
.stApp .hero p {
    color: #E7F0EA !important;
    font-size: 1.05rem;
    margin: 0;
    max-width: 640px;
}
.furrows {
    margin-top: 1.4rem;
    height: 6px;
    border-radius: 4px;
    background: repeating-linear-gradient(
        90deg,
        var(--wheat-light) 0px, var(--wheat-light) 22px,
        transparent 22px, transparent 30px
    );
    opacity: 0.85;
}

/* ---------- Feature label chips above each tab's content ---------- */
.chip {
    display: inline-block;
    padding: 0.28rem 0.85rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-bottom: 0.9rem;
}
.stApp .chip-green { background: #E4F1E8; color: var(--forest) !important; }
.stApp .chip-gold  { background: #FBEED9; color: var(--wheat) !important; }
.stApp .chip-blue  { background: #E4EEF3; color: var(--sky) !important; }

/* ---------- Buttons ---------- */
.stButton>button {
    background: var(--forest);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.6rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    transition: all 0.15s ease;
}
.stButton>button:hover {
    background: var(--forest-light);
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(27,67,50,0.25);
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #F1ECE0;
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    font-weight: 600;
    color: var(--soil);
}
.stTabs [aria-selected="true"] {
    background-color: var(--card) !important;
    color: var(--forest) !important;
    border-bottom: 3px solid var(--wheat) !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: #F1ECE0;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--forest) !important;
}

/* ---------- Result cards ---------- */
.result-card {
    padding: 1.3rem 1.5rem;
    border-radius: 12px;
    margin-top: 1rem;
    font-size: 1.05rem;
    font-weight: 600;
}
.stApp .result-green { background: #E4F1E8; color: var(--forest) !important; border-left: 5px solid var(--forest); }
.stApp .result-gold  { background: #FBEED9; color: #7A5210 !important; border-left: 5px solid var(--wheat); }
.stApp .result-blue  { background: #E4EEF3; color: #1F4A5C !important; border-left: 5px solid var(--sky); }
.stApp .result-card span.big { color: inherit !important; }
.result-card span.big { font-size: 1.5rem; display: block; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)


# ========================================================
# HERO BANNER
# ========================================================
st.markdown("""
<div class="hero">
    <h1>🌾 Smart Farmer Assistant</h1>
    <p>An ML-powered decision support tool that helps you pick the right crop,
    estimate your yield, and check tomorrow's weather — just enter your field's
    numbers and get an instant, data-driven answer.</p>
    <div class="furrows"></div>
</div>
""", unsafe_allow_html=True)


# ========================================================
# SIDEBAR
# ========================================================
with st.sidebar:
    st.markdown("### 🌱 About this app")
    st.write(
        "This project combines 3 machine learning models into one simple "
        "farming assistant. Every prediction is made instantly on your device "
        "— no data is sent anywhere."
    )
    st.markdown("---")
    st.markdown("### 🧠 Models used")
    st.markdown("""
    - **Crop Recommendation** — Random Forest
    - **Yield Prediction** — Random Forest Regressor
    - **Weather Insights** — Logistic Regression
    """)
    st.markdown("---")
    st.caption("Built with scikit-learn + Streamlit")


tab1, tab2, tab3 = st.tabs(["🌱  Crop Recommendation", "📈  Yield Prediction", "☁️  Weather Insights"])


# ======================================================
# TAB 1: CROP RECOMMENDATION
# ======================================================
with tab1:
    st.markdown('<span class="chip chip-green">RANDOM FOREST CLASSIFIER</span>', unsafe_allow_html=True)
    st.markdown("#### Find the best crop for your field")
    st.write("Enter your soil nutrients and local climate conditions below.")

    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Soil Nutrients**")
            N = st.slider("Nitrogen (N)", 0, 150, 50)
            P = st.slider("Phosphorus (P)", 0, 150, 50)
            K = st.slider("Potassium (K)", 0, 150, 50)
            ph = st.slider("Soil pH", 3.0, 10.0, 6.5)
        with col_b:
            st.markdown("**Climate**")
            temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
            rainfall = st.slider("Rainfall (mm)", 0.0, 350.0, 100.0)

        predict_crop = st.button("🔍 Recommend Crop", key="crop_btn")

    if predict_crop:
        input_data = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        )
        prediction = crop_model.predict(input_data)[0]
        st.markdown(f"""
        <div class="result-card result-green">
            ✅ Recommended crop for your field
            <span class="big">{prediction.upper()}</span>
        </div>
        """, unsafe_allow_html=True)


# ======================================================
# TAB 2: YIELD PREDICTION
# ======================================================
with tab2:
    st.markdown('<span class="chip chip-gold">RANDOM FOREST REGRESSOR</span>', unsafe_allow_html=True)
    st.markdown("#### Estimate your expected yield")
    st.write("Enter your farming conditions to get an estimated yield in tons per hectare.")

    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            rainfall2 = st.slider("Rainfall (mm)", 0, 350, 150, key="yield_rain")
            fertilizer = st.slider("Fertilizer used (kg/acre)", 0, 250, 100)
        with col_b:
            pesticide = st.slider("Pesticide used (kg/acre)", 0, 50, 10)
            temperature2 = st.slider("Temperature (°C)", 5, 45, 27, key="yield_temp")

        predict_yield = st.button("🔍 Predict Yield", key="yield_btn")

    if predict_yield:
        input_data = pd.DataFrame(
            [[rainfall2, fertilizer, pesticide, temperature2]],
            columns=["rainfall", "fertilizer", "pesticide", "temperature"]
        )
        prediction = yield_model.predict(input_data)[0]
        st.markdown(f"""
        <div class="result-card result-gold">
            ✅ Estimated yield
            <span class="big">{prediction:.2f} tons / hectare</span>
        </div>
        """, unsafe_allow_html=True)


# ======================================================
# TAB 3: WEATHER INSIGHTS
# ======================================================
with tab3:
    st.markdown('<span class="chip chip-blue">LOGISTIC REGRESSION</span>', unsafe_allow_html=True)
    st.markdown("#### Will it rain tomorrow?")
    st.write("Enter today's weather readings to forecast tomorrow's rain probability.")

    with st.container(border=True):
        col_a, col_b = st.columns(2)
        with col_a:
            humidity2 = st.slider("Humidity (%)", 0, 100, 65, key="weather_hum")
            pressure = st.slider("Pressure (hPa)", 985, 1030, 1010)
            wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)
        with col_b:
            temperature3 = st.slider("Temperature (°C)", 5, 45, 28, key="weather_temp")
            cloud_cover = st.slider("Cloud Cover (%)", 0, 100, 40)

        predict_rain = st.button("🔍 Predict Rain", key="rain_btn")

    if predict_rain:
        input_data = pd.DataFrame(
            [[humidity2, pressure, wind_speed, temperature3, cloud_cover]],
            columns=["humidity", "pressure", "wind_speed", "temperature", "cloud_cover"]
        )
        input_scaled = weather_scaler.transform(input_data)
        prediction = weather_model.predict(input_scaled)[0]
        probability = weather_model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card result-blue">
                🌧️ Rain expected tomorrow
                <span class="big">{probability*100:.1f}% probability</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-blue">
                ☀️ No rain expected tomorrow
                <span class="big">Only {probability*100:.1f}% rain probability</span>
            </div>
            """, unsafe_allow_html=True)
