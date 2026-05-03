import streamlit as st
import pandas as pd
import joblib


# ==================================
# LOAD SAVED FILES
# ==================================
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ==================================
# PAGE TITLE
# ==================================
st.title("📊 Customer Churn Prediction Dashboard")


# ==================================
# USER INPUTS
# ==================================
gender = st.selectbox("Gender", ["Male", "Female"])

tenure = st.number_input(
    "Tenure Months",
    min_value=0,
    max_value=100,
    value=12
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

monthly = st.number_input(
    "Monthly Charges",
    value=70.0
)

total = st.number_input(
    "Total Charges",
    value=1000.0
)


# ==================================
# PREDICT BUTTON
# ==================================
if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Tenure Months": [tenure],
        "Internet Service": [internet],
        "Tech Support": [tech_support],
        "Streaming TV": [streaming_tv],
        "Streaming Movies": [streaming_movies],
        "Contract": [contract],
        "Paperless Billing": [paperless],
        "Monthly Charges": [monthly],
        "Total Charges": [total]
    })

    # Encoding
    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )

    # Match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    # Result
    if prediction == 1:
        st.error(
            f"⚠ Customer may churn "
            f"({probability:.2%} risk)"
        )
    else:
        st.success(
            f"✅ Customer likely stays "
            f"({1-probability:.2%} confidence)"
        )
