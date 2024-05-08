import streamlit as st
import pandas as pd
import numpy as np
import joblib

brand_dic = {'Audi': 0, 'BMW': 1, 'Mercedes-Benz': 3, 'Renault': 4, 'Toyota': 5, 'Volkswagen': 6, 'Mitsubishi': 7}
body_dic = {'crossover': 0, 'hatch': 1, 'other': 2, 'sedan': 3, 'vagon': 4, 'van': 5}
engine_type_dic = {'Diesel': 0, 'Gas': 1, 'Other': 2, 'Petrol': 3}

st.set_page_config(page_title='Future Car Prediction')

car = pd.read_csv('Car_cleaned.csv')

def find_model(brand):
    model = car[car['Brand'] == brand]['Model']
    return list(model)

def model_loader(path):
    model = joblib.load(path)
    return model

model_forest = model_loader("rf1_base_rf.pkl")

st.markdown("<h2 style='text-align:center;'>Future Car Price Prediction</h2>", unsafe_allow_html=True)

brand_list = ['BMW', 'Mercedes-Benz', 'Audi', 'Toyota', 'Renault', 'Volkswagen', 'Mitsubishi']
body_list = ['sedan', 'van', 'crossover', 'vagon', 'other', 'hatch']
engine_type_list = ['Diesel', 'Gas', 'Other', 'Petrol']

brand_inp = st.selectbox(label='Select the brand of the car', options=brand_list)
brand = brand_dic[brand_inp]

if brand_inp == 'Audi':
    model_list = find_model('Audi')
elif brand_inp == 'Renault':
    model_list = find_model('Renault')
elif brand_inp == 'Toyota':
    model_list = find_model('Toyota')
elif brand_inp == 'BMW':
    model_list = find_model('BMW')
elif brand_inp == 'Mercedes-Benz':
    model_list = find_model('Mercedes-Benz')
elif brand_inp == 'Mitsubishi':
    model_list = find_model('Mitsubishi')
elif brand_inp == 'Volkswagen':
    model_list = find_model('Volkswagen')

model_inp = st.selectbox('Select the model of the car', options=model_list)

body_type = st.selectbox(label='Select the body type of the car', options=body_list)
body_type = body_dic[body_type]

engine_type = st.selectbox(label='Select the engine type (fuel)', options=engine_type_list)
engine_type = engine_type_dic[engine_type]

years_future = st.slider('Select the number of years in the future', 5, 15, 10)

predict = st.button('Predict')

if predict:
    inp_array = np.array([[0, 0, 0, brand, body_type, engine_type, 0, 0]])  # Placeholder values for mileage and year
    
    # Define model dictionary to map model names to numerical representations
    model_dic = {model: idx for idx, model in enumerate(model_list)}
    
    model_code = model_dic[model_inp]  # Find model code
    inp_array[0][6] = model_code  # Set model code in inp_array

    pred_current = model_forest.predict(inp_array)
    
    if pred_current < 0:
        st.error('The values seem to be irrelevant. Please provide relevant information and try again.')
    else:
        pred_current = round(float(pred_current), 3)

        # Assuming linear depreciation rate of 10% per year
        depreciation_rate = 0.1
        
        future_price = pred_current * (1 + depreciation_rate) ** years_future
        exchange_rate = 75  # Current exchange rate of 1 USD to INR

        # Convert predicted price from dollars to rupees
        future_price_inr = future_price * exchange_rate
        st.success(f"The predicted price of the car after {years_future} years is: ₹{round(future_price_inr, 3)}")
