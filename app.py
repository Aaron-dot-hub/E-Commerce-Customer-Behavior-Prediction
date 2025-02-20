import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
xgb_classifier = joblib.load("xgb_model.pkl")

# Title
st.title("E-commerce Customer Purchase Prediction")
st.write("Fill in the details below to predict whether the customer will purchase or not.")

# Create a form for input
with st.form("customer_data_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=26, max_value=43, step=1)
    city = st.selectbox("City", ["New York", "Los Angeles", "Chicago", "San Francisco", "Miami", "Houston"])
    membership_type = st.selectbox("Membership Type", ["Gold", "Silver", "Bronze"])
    total_spend = st.number_input("Total Spend", min_value=400,max_value=4000, step=400)
    items_purchased = st.number_input("Items Purchased", min_value=7,max_value=21, step=1)
    average_rating = st.slider("Average Rating", min_value=3.0, max_value=5.0, step=0.1)
    discount_applied = st.radio("Discount Applied?", ["Yes", "No"])
    days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=0, step=1)
    satisfaction_level = st.selectbox("Satisfaction Level", ["Satisfied", "Neutral", "Unsatisfied"])
  
    # Submit button
    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert inputs to the same format as your preprocessed data
    input_data = pd.DataFrame({
        "Gender": [1 if gender == "Male" else 0],
        "Age": [age],
        "City": [0 if city == "New York" else 1 if city == "Los Angeles" else 2 if city == "Chicago" else 3 if city == "San Francisco" else 4 if city == "Miami" else 5],
        "Membership Type": [2 if membership_type == "Gold" else 1 if membership_type == "Silver" else 0],
        "Total Spend": [total_spend],
        "Items Purchased": [items_purchased],
        "Average Rating": [average_rating],
        "Discount Applied": [1 if discount_applied == "Yes" else 0],
        "Days Since Last Purchase": [days_since_last_purchase],
        "Satisfaction Level": [0 if satisfaction_level == "Unsatisfied" else 1 if satisfaction_level == "Neutral" else 2],
    })

    # Ensure the input data column order matches the training data
    column_order = ["Gender", "Age", "City", "Membership Type", "Total Spend", "Items Purchased", "Average Rating", "Discount Applied", "Days Since Last Purchase", "Satisfaction Level"]
    input_data = input_data[column_order]

    # Display input data
    st.write("Input Data:", input_data)

    # Make prediction
    prediction = xgb_classifier.predict(input_data)[0]

    # Interpret the prediction
    result = "This Customer will  Purchase the Product" if prediction == 1 else "This Customer will not Purchase the Product"
    st.success(f"The model predicts: **{result}**")
