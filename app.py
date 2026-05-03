import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Churn Dashboard",
    layout="wide"
)

# LOAD FILES
model = joblib.load("data/churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

df = pd.read_excel("data/churn.xlsx")

df["Churn Value"] = df["Churn Value"].astype(int)

# TITLE
st.title("📊 Customer Churn Analytics Dashboard")

# KPI SECTION
total_customers = len(df)

churn_customers = df["Churn Value"].sum()

retention_rate = (
    (total_customers - churn_customers)
    / total_customers
) * 100

avg_revenue = df["Monthly Charges"].mean()


k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total Customers",
    total_customers
)

k2.metric(
    "Churn Customers",
    churn_customers
)

k3.metric(
    "Retention Rate",
    f"{retention_rate:.1f}%"
)

k4.metric(
    "Avg Monthly Revenue",
    f"${avg_revenue:.2f}"
)

st.divider()

# ATTRACTIVE CHARTS
c1, c2 = st.columns(2)

# DONUT CHART
churn_counts = (
    df["Churn Value"]
    .value_counts()
    .reset_index()
)

churn_counts.columns = [
    "Status",
    "Count"
]

churn_counts["Status"] = (
    churn_counts["Status"]
    .map({
        0: "Retained",
        1: "Churn"
    })
)

fig1 = px.pie(
    churn_counts,
    names="Status",
    values="Count",
    hole=0.6,
    title="Customer Retention Overview"
)

c1.plotly_chart(
    fig1,
    use_container_width=True
)

# LINE CHART
tenure_data = (
    df.groupby("Tenure Months")
    ["Monthly Charges"]
    .mean()
    .reset_index()
)

fig2 = px.line(
    tenure_data,
    x="Tenure Months",
    y="Monthly Charges",
    markers=True,
    title="Revenue Trend by Customer Tenure"
)

c2.plotly_chart(
    fig2,
    use_container_width=True
)

# SIDEBAR
st.sidebar.title("Predict Customer")


gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

tenure = st.sidebar.number_input(
    "Tenure Months",
    0,
    100,
    12
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

tech = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

monthly = st.sidebar.number_input(
    "Monthly Charges",
    0.0,
    200.0,
    70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

# PREDICTION
if st.sidebar.button("Predict"):

    input_df = pd.DataFrame({
        "Gender": [gender],
        "Tenure Months": [tenure],
        "Internet Service": [internet],
        "Tech Support": [tech],
        "Streaming TV": [tv],
        "Streaming Movies": [movies],
        "Contract": [contract],
        "Paperless Billing": [paperless],
        "Monthly Charges": [monthly],
        "Total Charges": [total]
    })


    # Encoding
    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )


    # Match training columns
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Scaling
    input_scaled = scaler.transform(
        input_df
    )


    # Prediction
    pred = model.predict(
        input_scaled
    )[0]


    prob = model.predict_proba(
        input_scaled
    )[0][1]


    st.divider()

    st.subheader(
        "🤖 Customer Prediction"
    )


    if pred == 1:

        st.error(
            f"⚠ High Churn Risk ({prob:.2%})"
        )

        revenue_risk = monthly * 12

        st.metric(
            "Annual Revenue At Risk",
            f"${revenue_risk:.0f}"
        )

        st.info(
            """
            Recommended Action:

            • Offer retention discount  
            • Assign customer support  
            • Upgrade service plan  
            """
        )

    else:

        stay_prob = 1 - prob

        st.success(
            f"✅ Customer Likely to Stay ({stay_prob:.2%})"
        )

        st.info(
            """
            Recommended Action:

            • Offer loyalty rewards  
            • Upsell premium services  
            • Promote long-term contracts  
            """
        )
