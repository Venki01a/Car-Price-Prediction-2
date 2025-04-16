import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Version check
st.sidebar.write(f"scikit-learn version: {sklearn.__version__}")

# Load the model and scaler with version compatibility check
@st.cache_resource
def load_model():
    try:
        with open('random_forest.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('random_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except AttributeError as e:
        st.error(f"Model loading error: {str(e)}")
        st.error("This is likely due to scikit-learn version mismatch.")
        st.error("Please ensure you're using scikit-learn >= 1.0.0")
        return None, None

model, scaler = load_model()

if model is None:
    st.stop()

# Create label encoders
model_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()

# Fit label encoders with possible values
model_encoder.fit(['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'Q2', 'Q3', 'Q5', 'Q7', 'Q8', 'R8', 'RS3', 'RS4', 'RS5', 'RS6', 'RS7', 'S3', 'S4', 'S5', 'S8', 'SQ5', 'SQ7', 'TT'])
fuel_encoder.fit(['Diesel', 'Hybrid', 'Petrol', 'Other'])

# Prediction function
def predict_price(input_data):
    try:
        # Preprocess the input data
        input_data[0] = model_encoder.transform([input_data[0]])[0]
        input_data[4] = fuel_encoder.transform([input_data[4]])[0]
        
        # One-hot encode transmission
        transmission_data = [[input_data[2]]]
        ct = ColumnTransformer(
            [('encoder', OneHotEncoder(categories=[['Manual', 'Automatic', 'Semi-Auto']]), [0])],
            remainder='passthrough'
        )
        encoded_transmission = ct.fit_transform(transmission_data)
        
        # Combine features
        processed_data = np.concatenate([
            encoded_transmission[0],
            [input_data[0]],
            [input_data[1]],
            [input_data[3]],
            [input_data[4]],
            [input_data[5]],
            [input_data[6]],
            [input_data[7]]
        ]).reshape(1, -1)
        
        # Scale and predict
        scaled_data = scaler.transform(processed_data)
        return model.predict(scaled_data)[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Streamlit UI
st.title('Car Price Prediction 2')
st.write('This app predicts the price of cars based on their features')

with st.form('car_features'):
    st.header('Enter Car Details')
    
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox('Model', model_encoder.classes_)
        year = st.slider('Year', 1997, 2020, 2017)
        transmission = st.selectbox('Transmission', ['Manual', 'Automatic', 'Semi-Auto'])
    with col2:
        mileage = st.number_input('Mileage', min_value=0, value=15000)
        fuel_type = st.selectbox('Fuel Type', fuel_encoder.classes_)
        tax = st.number_input('Tax (£)', min_value=0, value=145)
    
    col3, col4 = st.columns(2)
    with col3:
        mpg = st.number_input('MPG', min_value=0.0, value=50.0, step=0.1)
    with col4:
        engine_size = st.number_input('Engine Size', min_value=0.0, max_value=6.3, value=2.0, step=0.1)
    
    submitted = st.form_submit_button('Predict Price')

if submitted and model is not None:
    input_data = [
        model_name, year, transmission,
        mileage, fuel_type, tax,
        mpg, engine_size
    ]
    
    prediction = predict_price(input_data)
    if prediction is not None:
        st.success(f'Predicted Car Price: £{prediction:,.2f}')

# Sidebar information
st.sidebar.header('About')
st.sidebar.info("""
This prediction model uses a Random Forest Regressor trained on Audi car data.
- **Accuracy**: 98.6%
- **Mean Absolute Error**: £784
""")