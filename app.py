import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan preprocessor
model = joblib.load('best_churn_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title("Aplikasi Prediksi Churn Pelanggan 📊")
st.write("Masukkan data pelanggan di bawah ini untuk melihat prediksi.")

# Buat form input sesuai dengan fitur utama yang digunakan di dataset Anda
# Contoh input numerik dan kategorikal:
age = st.number_input("Usia (Age)", min_value=1, max_value=100, value=35)
total_visits = st.number_input("Total Kunjungan", min_value=0, value=15)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
# ... tambahkan fitur penting lainnya dari dataset ...

if st.button("Prediksi"):
    # Satukan input menjadi DataFrame berbentuk 1 baris
    input_data = pd.DataFrame([{
        'age': age,
        'total_visits': total_visits,
        'gender': gender,
        # ... sesuaikan nama kolom persis seperti df awal sebelum preprocessing ...
    }])
    
    # Transformasi data menggunakan preprocessor bawaan dari training
    input_preprocessed = preprocessor.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_preprocessed)
    
    if prediction[0] == 1:
        st.error("Pelanggan berpotensi CHURN (Berhenti Berlangganan) 🔴")
    else:
        st.success("Pelanggan Tetap AKTIF (Setia) 🟢")