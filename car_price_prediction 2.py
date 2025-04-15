import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('random_forest.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('random_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Dummy encoders – replace with actual mappings from your dataset
model_encoder = {
    'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3, 'A5':4,
    'A6':5,'A7':6, 'A8': 7, 'Q2':8, 'Q3':9, 'Q5':10,
    'Q7':11, 'Q8':12, 'R8':13, 'RS3':14, 'RS4':15,
    'RS5':16, 'RS6':17, 'RS7':18, 'S3':19, 'S4':20,
    'S5':21, 'S8':22, 'SQ5':23, 'SQ7':24, 'TT':25
}

transmission_encoder = {
    'Manual': 0,
    'Automatic': 1,
    'Semi-Auto': 2
}

fuel_encoder = {
    'Petrol': [1, 0, 0],
    'Diesel': [0, 1, 0],
    'Hybrid': [0, 0, 1]
}

st.title("🚗 Car Price Prediction App")

# Input Fields
model_name = st.selectbox("Car Model", list(model_encoder.keys()))
year = st.number_input("Year", min_value=1990, max_value=2025, step=1)
transmission = st.selectbox("Transmission", list(transmission_encoder.keys()))
fuel_type = st.selectbox("Fuel Type", list(fuel_encoder.keys()))
tax = st.number_input("Tax (₹)", min_value=0)
mpg = st.number_input("Mileage per Gallon (mpg)", min_value=0.0)
engine_size = st.number_input("Engine Size (L)", min_value=0.0)
mileage = st.number_input("Mileage (in km)", min_value=0)

# Predict button
if st.button("Predict Price"):
    model_val = model_encoder[model_name]
    transmission_val = transmission_encoder[transmission]
    fuel_vals = fuel_encoder[fuel_type]

    # Final input list – match training order
    input_data = [model_val, year] + fuel_vals + [tax, mpg, engine_size, transmission_val, mileage]

    # Scale input
    input_scaled = scaler.transform([input_data])

    # Prediction
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Selling Price: ₹ {prediction:,.2f}")

