import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# --------------------------------
# Load model and threshold
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/credit_risk_model.pkl")


@st.cache_resource
def load_threshold():
    return joblib.load("models/threshold.pkl")


model = load_model()
threshold = load_threshold()


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

st.title("Credit Risk Prediction")

st.write(
    "Estimate a borrower's probability of default"
)


# --------------------------------
# Borrower Information
# --------------------------------

st.header("Borrower Information")

revenue = st.number_input(
    "Revenue",
    min_value=0.0,
    value=60000.0,
    step=1000.0
)

dti_n = st.number_input(
    "Debt-to-Income Ratio",
    min_value=0.0,
    value=17.0,
    step=0.1
)

loan_amnt = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=10000.0,
    step=500.0
)

fico_n = st.number_input(
    "FICO Score",
    min_value=300.0,
    max_value=850.0,
    value=680.0,
    step=1.0
)

experience_c = st.selectbox(
    "New Customer",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

emp_length = st.selectbox(
    "Employment Length",
    options=[
        "< 1 year",
        "1 year",
        "2 years",
        "3 years",
        "4 years",
        "5 years",
        "6 years",
        "7 years",
        "8 years",
        "9 years",
        "10+ years",
        "NI"
    ]
)

purpose = st.selectbox(
    "Loan Purpose",
    options=[
        "credit_card",
        "debt_consolidation",
        "other",
        "small_business",
        "medical",
        "moving",
        "vacation",
        "car"
    ]
)

home_ownership_n = st.selectbox(
    "Home Ownership",
    options=[
        "MORTGAGE",
        "RENT",
        "OWN"
    ]
)


# --------------------------------
# Feature Engineering
# --------------------------------

if revenue > 0:
    loan_to_income = loan_amnt / revenue
else:
    loan_to_income = np.nan


# --------------------------------
# Create input DataFrame
# --------------------------------

input_data = pd.DataFrame({
    "revenue": [revenue],
    "dti_n": [dti_n],
    "loan_amnt": [loan_amnt],
    "fico_n": [fico_n],
    "experience_c": [experience_c],
    "loan_to_income": [loan_to_income],
    "emp_length": [emp_length],
    "purpose": [purpose],
    "home_ownership_n": [home_ownership_n]
})


# --------------------------------
# Prediction
# --------------------------------

if st.button("Predict Default Risk"):

    # ----------------------------
    # Probability of Default
    # ----------------------------

    pd_probability = model.predict_proba(input_data)[0, 1]

    prediction = int(pd_probability >= threshold)


    # ----------------------------
    # Prediction Result
    # ----------------------------

    st.subheader("Prediction Result")

    st.metric(
        "Probability of Default",
        f"{pd_probability:.2%}"
    )

    if prediction == 1:
        st.error("Higher Risk of Default")
    else:
        st.success("Lower Risk of Default")

    st.write(
        f"Decision threshold: {threshold:.0%}"
    )


# --------------------------------
# SHAP Explanation
# --------------------------------

st.subheader("Why did the model make this prediction?")

st.caption(
    "SHAP identifies the features that pushed this borrower's "
    "underlying XGBoost risk score higher or lower."
)



# -------------------------------
# Extract XGBoost model
# --------------------------------

xgb_pipeline = model.calibrated_classifiers_[0].estimator

preprocessor = xgb_pipeline.named_steps["preprocessor"]
xgb_model = xgb_pipeline.named_steps["model"]


# --------------------------------
# Transform borrower input
# --------------------------------

input_transformed = preprocessor.transform(input_data)
input_dense = input_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()


# --------------------------------
# Calculate SHAP values
# --------------------------------

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(input_dense)


# --------------------------------
# Create SHAP DataFrame
# --------------------------------

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": shap_values[0],
    "Feature Value": input_dense[0]
})


# --------------------------------
# Remove inactive one-hot features
# --------------------------------

shap_df = shap_df[
    shap_df["Feature Value"] != 0
].copy()


# --------------------------------
# Readable feature names
# --------------------------------

def make_readable_name(feature):

    feature = feature.replace("num__", "")
    feature = feature.replace("cat__", "")

    replacements = {
        "loan_to_income": "Loan-to-income ratio",
        "fico_n": "FICO score",
        "dti_n": "Debt-to-income ratio",
        "loan_amnt": "Loan amount",
        "revenue": "Revenue",
        "experience_c": "New customer",
    }

    if feature in replacements:
        return replacements[feature]

    if feature.startswith("emp_length_"):
        return "Employment: " + feature.replace(
            "emp_length_", ""
        )

    if feature.startswith("purpose_"):
        return "Purpose: " + feature.replace(
            "purpose_", ""
        )

    if feature.startswith("home_ownership_n_"):
        return "Home ownership: " + feature.replace(
            "home_ownership_n_", ""
        )

    return feature


shap_df["Feature"] = shap_df["Feature"].apply(
    make_readable_name
)


# --------------------------------
# Separate risk-increasing factors
# --------------------------------

risk_increasing = (
    shap_df[shap_df["SHAP Value"] > 0]
    .sort_values("SHAP Value", ascending=False)
    .head(5)
)


# --------------------------------
# Separate risk-reducing factors
# --------------------------------

risk_reducing = (
    shap_df[shap_df["SHAP Value"] < 0]
    .sort_values("SHAP Value", ascending=True)
    .head(5)
)


# --------------------------------
# Display risk-increasing factors
# --------------------------------

st.write("### Top factors increasing risk")

if len(risk_increasing) > 0:

    for feature in risk_increasing["Feature"]:
        st.write(f"• {feature}")

else:

    st.write("No major risk-increasing factors identified.")


# --------------------------------
# Display risk-reducing factors
# --------------------------------

st.write("### Factors reducing risk")

if len(risk_reducing) > 0:

    for feature in risk_reducing["Feature"]:
        st.write(f"• {feature}")

else:

    st.write("No major risk-reducing factors identified.")


# --------------------------------
# SHAP Bar Chart
# --------------------------------

st.write("### SHAP feature contributions")

chart_df = pd.concat([
    risk_increasing,
    risk_reducing
]).copy()

chart_df = chart_df.sort_values(
    "SHAP Value"
)

chart_df = chart_df[
    ["Feature", "SHAP Value"]
]

st.bar_chart(
    chart_df.set_index("Feature")
)