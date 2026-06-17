import streamlit as st
import pandas as pd
import joblib

# Load model dan preprocessor
model = joblib.load('best_churn_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title("Aplikasi Prediksi Churn Pelanggan 📊")
st.write("Masukkan data pelanggan di bawah ini untuk melihat prediksi.")

# ===== Data Demografi =====
st.subheader("Data Demografi")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Usia", min_value=1, max_value=100, value=35)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    country = st.text_input("Negara", value="Indonesia")
with col2:
    city = st.text_input("Kota", value="Jakarta")
    is_premium_user = st.selectbox("Premium User?", [0, 1])

# ===== Aktivitas =====
st.subheader("Aktivitas Pelanggan")
col1, col2 = st.columns(2)
with col1:
    total_visits = st.number_input("Total Kunjungan", min_value=0, value=15)
    avg_session_time = st.number_input("Rata-rata Durasi Sesi (menit)", min_value=0.0, value=10.0)
    pages_per_session = st.number_input("Halaman per Sesi", min_value=0.0, value=5.0)
with col2:
    email_open_rate = st.number_input("Email Open Rate (0-1)", min_value=0.0, max_value=1.0, value=0.5)
    email_click_rate = st.number_input("Email Click Rate (0-1)", min_value=0.0, max_value=1.0, value=0.2)
    last_3_month_purchase_freq = st.number_input("Frekuensi Belanja 3 Bulan Terakhir", min_value=0, value=3)

# ===== Transaksi =====
st.subheader("Data Transaksi")
col1, col2 = st.columns(2)
with col1:
    total_spent = st.number_input("Total Belanja", min_value=0.0, value=500.0)
    avg_order_value = st.number_input("Rata-rata Nilai Order", min_value=0.0, value=50.0)
    discount_used = st.number_input("Jumlah Diskon Digunakan", min_value=0, value=2)
with col2:
    lifetime_value = st.number_input("Lifetime Value", min_value=0.0, value=1000.0)
    marketing_spend_per_user = st.number_input("Marketing Spend per User", min_value=0.0, value=20.0)

# ===== Layanan =====
st.subheader("Layanan & Pengiriman")
col1, col2 = st.columns(2)
with col1:
    support_tickets = st.number_input("Jumlah Support Tickets", min_value=0, value=1)
    refund_requested = st.number_input("Jumlah Refund Diminta", min_value=0, value=0)
    delivery_delay_days = st.number_input("Rata-rata Keterlambatan Pengiriman (hari)", min_value=0, value=0)
with col2:
    satisfaction_score = st.number_input("Skor Kepuasan (1-10)", min_value=1, max_value=10, value=7)
    nps_score = st.number_input("NPS Score (0-10)", min_value=0, max_value=10, value=7)

# ===== Channel & Subscription =====
st.subheader("Channel & Subscription")
col1, col2 = st.columns(2)
with col1:
    acquisition_channel = st.selectbox("Acquisition Channel", ["Organic", "Paid Ads", "Referral", "Social Media", "Email"])
    device_type = st.selectbox("Tipe Device", ["Mobile", "Desktop", "Tablet"])
with col2:
    subscription_type = st.selectbox("Tipe Subscription", ["Free", "Basic", "Premium"])
    payment_method = st.selectbox("Metode Pembayaran", ["Credit Card", "Debit Card", "E-Wallet", "Bank Transfer"])

if st.button("Prediksi", type="primary"):
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'country': country,
        'city': city,
        'acquisition_channel': acquisition_channel,
        'device_type': device_type,
        'subscription_type': subscription_type,
        'is_premium_user': is_premium_user,
        'total_visits': total_visits,
        'avg_session_time': avg_session_time,
        'pages_per_session': pages_per_session,
        'email_open_rate': email_open_rate,
        'email_click_rate': email_click_rate,
        'total_spent': total_spent,
        'avg_order_value': avg_order_value,
        'discount_used': discount_used,
        'support_tickets': support_tickets,
        'refund_requested': refund_requested,
        'delivery_delay_days': delivery_delay_days,
        'payment_method': payment_method,
        'satisfaction_score': satisfaction_score,
        'nps_score': nps_score,
        'marketing_spend_per_user': marketing_spend_per_user,
        'lifetime_value': lifetime_value,
        'last_3_month_purchase_freq': last_3_month_purchase_freq,
    }])

    input_preprocessed = preprocessor.transform(input_data)
    prediction = model.predict(input_preprocessed)

    # Tampilkan probabilitas kalau ada
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_preprocessed)[0]
        st.write(f"Probabilitas Churn: **{proba[1]*100:.2f}%**")

    if prediction[0] == 1:
        st.error("Pelanggan berpotensi CHURN (Berhenti Berlangganan) 🔴")
    else:
        st.success("Pelanggan Tetap AKTIF (Setia) 🟢")
