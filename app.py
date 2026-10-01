import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load model, scaler and columns
# -----------------------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's details below and click **Predict**.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"]
    )
    resting_bp = st.number_input(
        "Resting Blood Pressure",
        50,
        250,
        120
    )
    cholesterol = st.number_input(
        "Cholesterol",
        0,
        700,
        200
    )
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120",
        [0, 1]
    )

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["LVH", "Normal", "ST"]
    )
    max_hr = st.number_input(
        "Maximum Heart Rate",
        60,
        220,
        150
    )
    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["N", "Y"]
    )
    oldpeak = st.number_input(
        "Oldpeak",
        -5.0,
        10.0,
        1.0,
        step=0.1
    )
    st_slope = st.selectbox(
        "ST Slope",
        ["Down", "Flat", "Up"]
    )

if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Age": [age],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "MaxHR": [max_hr],
        "Oldpeak": [oldpeak],

        "Sex_M": [1 if sex == "M" else 0],

        "ChestPainType_ATA": [1 if chest_pain == "ATA" else 0],
        "ChestPainType_NAP": [1 if chest_pain == "NAP" else 0],
        "ChestPainType_TA": [1 if chest_pain == "TA" else 0],

        "RestingECG_Normal": [1 if resting_ecg == "Normal" else 0],
        "RestingECG_ST": [1 if resting_ecg == "ST" else 0],

        "ExerciseAngina_Y": [1 if exercise_angina == "Y" else 0],

        "ST_Slope_Flat": [1 if st_slope == "Flat" else 0],
        "ST_Slope_Up": [1 if st_slope == "Up" else 0]
    })

    # Match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")