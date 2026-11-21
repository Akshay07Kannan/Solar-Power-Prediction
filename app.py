import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Solar Power Predictor', page_icon='☀️', layout='centered')

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'saved_models' / 'solar_power_model.joblib'

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title('☀️ Solar Power Generation Prediction')
st.write('Predict total AC power output using solar plant weather conditions and time.')

if not MODEL_PATH.exists():
    st.error('Model file not found. Put solar_power_model.joblib inside the saved_models folder.')
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f'Could not load the trained model: {exc}')
    st.stop()

plant_name = st.selectbox('Select solar plant', ['Plant 1', 'Plant 2'])
plant_id = {'Plant 1': '4135001', 'Plant 2': '4136001'}[plant_name]

col1, col2 = st.columns(2)
with col1:
    ambient = st.number_input('Ambient temperature (°C)', value=28.0, step=0.5)
    irradiation = st.number_input('Irradiation (dataset units)', min_value=0.0, value=0.65, step=0.05, format='%.3f')
with col2:
    module = st.number_input('Module temperature (°C)', value=35.0, step=0.5)
    date_value = st.date_input('Prediction date', value=dt.date(2020, 6, 1))

time_value = st.time_input('Prediction time', value=dt.time(12, 0))

if st.button('Predict Solar Power', type='primary', use_container_width=True):
    hour = time_value.hour + time_value.minute / 60 + time_value.second / 3600
    day_of_year = date_value.timetuple().tm_yday
    row = pd.DataFrame([{
        'AMBIENT_TEMPERATURE': ambient,
        'MODULE_TEMPERATURE': module,
        'IRRADIATION': irradiation,
        'HOUR_SIN': np.sin(2 * np.pi * hour / 24),
        'HOUR_COS': np.cos(2 * np.pi * hour / 24),
        'DAY_SIN': np.sin(2 * np.pi * day_of_year / 366),
        'DAY_COS': np.cos(2 * np.pi * day_of_year / 366),
        'PLANT_ID': plant_id,
    }])
    try:
        prediction = float(model.predict(row)[0])
        st.metric('Predicted total AC power (dataset units)', f'{max(0.0, prediction):,.2f}')
        st.caption('Negative model predictions are displayed as zero because physical power output cannot be negative. Units depend on the source dataset; do not assume kW without verification.')
        with st.expander('Show model input features'):
            st.dataframe(row, use_container_width=True)
    except Exception as exc:
        st.error(f'Prediction failed: {exc}')

st.divider()
st.caption('Model: Histogram Gradient Boosting Regression | Historical evaluation R²: 0.9637 | Predictions are estimates, not guarantees.')
