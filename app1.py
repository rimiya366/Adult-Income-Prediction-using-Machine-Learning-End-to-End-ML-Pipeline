import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Adult Income Predictor",
    page_icon="💼",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("ℹ️ About This App")
st.sidebar.write(
    "This AI app predicts whether a person's annual income is **greater than $50K or less than or equal to $50K** "
    "based on demographic and work-related features."
)

st.sidebar.subheader("🤖 Model Information")
st.sidebar.success("Machine Learning Model:XGBoost ")

st.sidebar.subheader("📊 Features Used")
st.sidebar.write("""
- Age  
- Workclass  
- Education  
- Marital Status  
- Occupation  
- Relationship  
- Race  
- Gender  
- Capital Gain / Loss  
- Hours per Week  
- Native Country
""")

st.sidebar.subheader("💡 Usage")
st.sidebar.write("Enter the details and click **Predict Income** to see the prediction.")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title{
    font-size:40px;
    font-weight:700;
    text-align:center;
    color:#2E86C1;
}
.sub-text{
    text-align:center;
    font-size:18px;
}
.result-box{
    padding:20px;
    border-radius:10px;
    background-color:#F4F6F7;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("adult_income.pkl")
scaler = joblib.load("scaler.pkl")

label1=joblib.load("le1.pkl")
label3=joblib.load("le3.pkl")
label4=joblib.load("le4.pkl")
label5=joblib.load("le5.pkl")
label6=joblib.load("le6.pkl")
label7=joblib.load("le7.pkl")
label8=joblib.load("le8.pkl")
label9=joblib.load("le9.pkl")

# ---------------- TITLE ----------------
st.markdown('<p class="main-title">💼 Adult Income Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Predict whether a person earns more than 50K per year</p>', unsafe_allow_html=True)

st.divider()

# ---------------- INPUT LAYOUT ----------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 75, 30)
    workclass = st.selectbox("Workclass", label9.classes_)
    education = st.selectbox("Education", label8.classes_)
    educational_num = st.number_input("Educational Number", 1, 16, 10)

with col2:
    marital_status = st.selectbox("Marital Status", label7.classes_)
    occupation = st.selectbox("Occupation", label6.classes_)
    relationship = st.selectbox("Relationship", label5.classes_)
    race = st.selectbox("Race", label4.classes_)

with col3:
    gender = st.selectbox("Gender", label1.classes_)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
    hours_per_week = st.number_input("Hours per Week", 1, 100, 40)
    native_country = st.selectbox("Native Country", label3.classes_)

st.divider()

# ---------------- ENCODING ----------------
gender= label1.transform([gender])[0]
native_country=label3.transform([native_country])[0]
race=label4.transform([race])[0]
relationship=label5.transform([relationship])[0]
occupation=label6.transform([occupation])[0]
marital_status=label7.transform([marital_status])[0]
education=label8.transform([education])[0]
workclass=label9.transform([workclass])[0]

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Income", use_container_width=True):

    input_data = np.array([[age,
                            workclass,
                            education,
                            educational_num,
                            marital_status,
                            occupation,
                            relationship,
                            race,
                            gender,
                            capital_gain,
                            capital_loss,
                            hours_per_week,
                            native_country]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
# -------- INPUT REVIEW BOX --------
    st.subheader("📋 Input Review")

    review_data = {
        "Feature": [
            "Age","Workclass","Education","Educational Number",
            "Marital Status","Occupation","Relationship",
            "Race","Gender","Capital Gain","Capital Loss",
            "Hours per Week","Native Country"
        ],
        "Value": [
            age, workclass, education, educational_num,
            marital_status, occupation, relationship,
            race, gender, capital_gain, capital_loss,
            hours_per_week, native_country
        ]
    }
    st.table(review_data)

    st.divider()
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if prediction == 1:
        st.success("💰 Predicted Income: >50K")
    else:
        st.warning("💼 Predicted Income: <=50K")

    st.info(f"📊 Probability of earning >50K: {round(probability*100,2)}%")

    st.markdown('</div>', unsafe_allow_html=True)
