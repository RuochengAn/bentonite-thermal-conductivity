import joblib
import xgboost as xgb
import numpy as np
import streamlit as st

import streamlit as st

st.image('JLU.png')


st.header('ML model for predicting thermal conductivity of bentonite')
st.caption('The used ML model is XGB')

model = joblib.load('ML model.joblib')
ss = joblib.load('StandardScaler.joblib')

col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input(u'$\mathrm{Temperature\;(℃)}$', step=0.01, format='%.2f')
    feature2 = st.number_input(u'$\mathrm{Water\;content\;(\%)}$', step=0.01, format='%.2f')
    feature3 = st.number_input(u'$\mathrm{Dry\;density\;(g/cm^3)}$', step=0.01, format='%.2f')
    feature4 = st.number_input(u'$\mathrm{Montmorillonite\;content\;(\%)}$', step=0.01, format='%.2f')

with col2:
    feature5 = st.number_input(u'$\mathrm{Liquid\;limit\;(\%)}$', step=0.01, format='%.2f')
    feature6 = st.number_input(u'$\mathrm{Plastic\;limit\;(\%)}$', step=0.01, format='%.2f')
    feature7 = st.number_input(u'$\mathrm{Graphite\;content\;(\%)}$', step=0.01, format='%.2f')
    feature8 = st.number_input(u'$\mathrm{Sand\;content\;(\%)}$', step=0.01, format='%.2f')
    
feature_values = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]

if st.button('Predict', type='primary'):
    input_data = np.array([feature_values])
    input_data_scaled = ss.transform(input_data)
    prediction = model.predict(input_data_scaled)
    st.success(f'Predicted thermal conductivity: {prediction[0]:.2f} W/mK', icon="✅")
