
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Illinois Mortgage Approval Predictor",
    page_icon="",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    * { font-family: Poppins, sans-serif; }
    .main { background-color: #F0F2F5; }
    .stApp { background-color: #F0F2F5; }
    h1 { color: #1B2A4A; font-weight: 700; }
    h2, h3 { color: #1B2A4A; font-weight: 600; }
    .result-approved {
        background-color: #1B2A4A;
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .result-denied {
        background-color: #F5A623;
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .disclaimer {
        background-color: #E8ECF2;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        color: #8892A4;
        margin-top: 2rem;
    }
    .section-header {
        background-color: #1B2A4A;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model         = joblib.load('model/xgb_model.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    threshold     = joblib.load('model/threshold.pkl')
    median_values = joblib.load('model/median_values.pkl')
    return model, feature_names, threshold, median_values

model, feature_names, threshold, median_values = load_model()

st.title("Illinois Mortgage Approval Predictor")
st.markdown("""
This tool uses a machine learning model trained on 119,663 real mortgage 
applications from Illinois (HMDA 2024) to estimate the likelihood of 
mortgage approval or denial.

This is a research tool for educational purposes only. 
It does not constitute financial advice.
""")

st.divider()

st.markdown('<div class="section-header">Financial Information</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    income = st.number_input(
        "Annual Income (thousands USD)",
        min_value=0, max_value=10000,
        value=80, step=5
    )
    loan_amount = st.number_input(
        "Loan Amount (USD)",
        min_value=10000, max_value=2000000,
        value=250000, step=10000
    )
    property_value = st.number_input(
        "Property Value (USD)",
        min_value=10000, max_value=3000000,
        value=300000, step=10000
    )

with col2:
    debt_to_income_ratio = st.slider(
        "Debt-to-Income Ratio",
        min_value=0, max_value=60, value=36
    )
    loan_to_value_ratio = st.slider(
        "Loan-to-Value Ratio (%)",
        min_value=10, max_value=120, value=90
    )
    loan_term = st.selectbox(
        "Loan Term (months)",
        options=[180, 240, 360],
        index=2,
        format_func=lambda x: f"{x} months ({x//12} years)"
    )

st.markdown('<div class="section-header">Property Information</div>',
            unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    loan_type = st.selectbox(
        "Loan Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Conventional",
            2: "FHA",
            3: "VA",
            4: "USDA"
        }[x]
    )
    construction_method = st.selectbox(
        "Construction Method",
        options=[1, 2],
        format_func=lambda x: {
            1: "Site-built",
            2: "Manufactured Home"
        }[x]
    )

with col4:
    total_units = st.selectbox(
        "Number of Units",
        options=[1, 2, 3, 4],
        index=0
    )
    submission_of_application = st.selectbox(
        "Application Submitted Via",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Broker",
            2: "Direct to institution",
            3: "Other"
        }[x]
    )

st.markdown('<div class="section-header">Applicant Demographics</div>',
            unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    derived_race = st.selectbox(
        "Race",
        options=[
            "White",
            "Black or African American",
            "Asian",
            "American Indian or Alaska Native",
            "Native Hawaiian or Other Pacific Islander",
            "2 or more minority races",
            "Joint"
        ]
    )
    derived_ethnicity = st.selectbox(
        "Ethnicity",
        options=[
            "Not Hispanic or Latino",
            "Hispanic or Latino",
            "Joint"
        ]
    )

with col6:
    derived_sex = st.selectbox(
        "Sex",
        options=["Male", "Female", "Joint"]
    )
    applicant_age = st.selectbox(
        "Age Group",
        options=["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
    )

st.markdown('<div class="section-header">Neighborhood Context</div>',
            unsafe_allow_html=True)

col7, col8 = st.columns(2)

with col7:
    tract_minority_population_percent = st.slider(
        "Minority Population in Neighborhood (%)",
        min_value=0, max_value=100, value=20
    )

with col8:
    tract_to_msa_income_percentage = st.slider(
        "Neighborhood Income vs. Metro Average (%)",
        min_value=0, max_value=200, value=100
    )

st.divider()

if st.button("Predict Approval Likelihood", use_container_width=True):

    input_data = median_values.copy()

    input_data["income"]                            = income
    input_data["loan_amount"]                       = loan_amount
    input_data["property_value"]                    = property_value
    input_data["debt_to_income_ratio"]              = debt_to_income_ratio
    input_data["loan_to_value_ratio"]               = loan_to_value_ratio
    input_data["loan_term"]                         = loan_term
    input_data["loan_type"]                         = loan_type
    input_data["construction_method"]               = construction_method
    input_data["total_units"]                       = total_units
    input_data["submission_of_application"]         = submission_of_application
    input_data["tract_minority_population_percent"] = tract_minority_population_percent
    input_data["tract_to_msa_income_percentage"]    = tract_to_msa_income_percentage

    for col in feature_names:
        if col.startswith("derived_race_"):
            input_data[col] = 0
    race_col = "derived_race_" + derived_race.replace(" ", "_").replace("-", "_")
    if race_col in input_data:
        input_data[race_col] = 1

    for col in feature_names:
        if col.startswith("derived_ethnicity_"):
            input_data[col] = 0
    eth_col = "derived_ethnicity_" + derived_ethnicity.replace(" ", "_").replace("-", "_")
    if eth_col in input_data:
        input_data[eth_col] = 1

    for col in feature_names:
        if col.startswith("derived_sex_"):
            input_data[col] = 0
    sex_col = "derived_sex_" + derived_sex.replace(" ", "_")
    if sex_col in input_data:
        input_data[sex_col] = 1

    for col in feature_names:
        if col.startswith("applicant_age_"):
            input_data[col] = 0
    age_clean = applicant_age.replace("<", "_").replace(">", "_").replace("-", "_")
    age_col = "applicant_age_" + age_clean
    if age_col in input_data:
        input_data[age_col] = 1

    input_df = pd.DataFrame([input_data])[feature_names]

    prob_denied   = model.predict_proba(input_df)[0][1]
    prob_approved = 1 - prob_denied
    prediction    = int(prob_denied >= threshold)

    if prediction == 0:
        st.markdown(f"""
        <div class="result-approved">
            Likely Approved<br>
            <span style="font-size:0.9rem; font-weight:300">
            Estimated approval probability: {prob_approved*100:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-denied">
            Likely Denied<br>
            <span style="font-size:0.9rem; font-weight:300">
            Estimated denial probability: {prob_denied*100:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Probability Breakdown")
    col_a, col_b = st.columns(2)
    col_a.metric("Approval Probability", f"{prob_approved*100:.1f}%")
    col_b.metric("Denial Probability",   f"{prob_denied*100:.1f}%")
    st.progress(float(prob_approved))

st.markdown("""
<div class="disclaimer">
    This tool is built for educational and research purposes only, based on 
    HMDA 2024 data for Illinois. It does not constitute financial or legal advice.
    The model was trained on publicly available data and may not reflect the 
    criteria used by any specific lender. In the United States, lenders are 
    prohibited from making credit decisions based on race, ethnicity, sex, or age 
    under the Fair Housing Act and Equal Credit Opportunity Act.
</div>
""", unsafe_allow_html=True)
