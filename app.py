import streamlit as st
import pandas as pd
import joblib

# Load the scaler and models
scaler = joblib.load('scaler.joblib')
ada_boost_model = joblib.load('ada_boost_model.joblib')
gradient_boosting_model = joblib.load('gradient_boosting_model.joblib')
xgb_model = joblib.load('xgb_model.joblib')

# Feature names (ensure these match the training order)
feature_names = ['ParentalSupport', 'Extracurricular', 'GPA', 'Tutoring', 'Absences', 'Gender', 'ParentalEducation', 'StudyTimeWeekly']

# GradeClass mapping
label_map = {
    0: "Sangat Rendah",
    1: "Rendah",
    2: "Sedang",
    3: "Baik",
    4: "Sangat Baik"
}

st.set_page_config(page_title="Student Academic Performance Predictor", layout="wide")
st.title("🎓 Prediksi Performa Akademik Mahasiswa")

st.write("### Masukkan Data Mahasiswa untuk Prediksi")

# Create a sidebar for model selection
st.sidebar.header("Pilih Model")
selected_model_name = st.sidebar.selectbox(
    "Pilih Algoritma Model:",
    ("AdaBoost", "Gradient Boosting", "XGBoost")
)

models = {
    "AdaBoost": ada_boost_model,
    "Gradient Boosting": gradient_boosting_model,
    "XGBoost": xgb_model
}

selected_model = models[selected_model_name]

# Input fields for features
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        parental_support = st.slider('Dukungan Orang Tua (0=Tidak ada, 4=Sangat Tinggi)', 0, 4, 2)
        extracurricular = st.radio('Kegiatan Ekstrakurikuler (0=Tidak, 1=Ya)', (0, 1))
        gpa = st.number_input('IPK (0.0 - 4.0)', min_value=0.0, max_value=4.0, value=2.5, step=0.01)
    
    with col2:
        tutoring = st.radio('Bimbingan Belajar (0=Tidak, 1=Ya)', (0, 1))
        absences = st.slider('Jumlah Ketidakhadiran', 0, 30, 10)
        gender = st.radio('Jenis Kelamin (0=Perempuan, 1=Laki-laki)', (0, 1))
    
    with col3:
        parental_education = st.slider('Pendidikan Orang Tua (0=Tidak ada, 3=Tinggi)', 0, 3, 1)
        study_time_weekly = st.number_input('Waktu Belajar Mingguan (jam)', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    
    submit_button = st.form_submit_button("Prediksi")

if submit_button:
    # Create a DataFrame from inputs
    input_data = pd.DataFrame([[parental_support, extracurricular, gpa, tutoring, absences, gender, parental_education, study_time_weekly]], columns=feature_names)
    
    # Scale the input data
    scaled_input_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = selected_model.predict(scaled_input_data)[0]
    
    st.subheader("Hasil Prediksi:")
    st.success(f"GradeClass yang Diprediksi: **{int(prediction)}**")
    st.info(f"Interpretasi: **{label_map[prediction]}**")

st.write("---_Note: The 'GradeClass' indicates academic performance from 0 (Very Low) to 4 (Very Good)._---")
