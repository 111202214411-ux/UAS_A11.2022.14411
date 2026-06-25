import streamlit as st
import pandas as pd
import joblib

# 1. Konfigurasi Halaman Dasar
st.set_page_config(page_title="Customer Retention Analysis", layout="centered")

# 2. Menyembunyikan menu bawaan Streamlit agar terlihat seperti Web App kustom
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load model dan preprocessor (ditambah cache agar lebih cepat saat reload)
@st.cache_resource
def load_objects():
    model = joblib.load('best_churn_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_objects()

# Judul dan Deskripsi yang lebih profesional
st.title("Sistem Analisis Retensi Pelanggan")
st.markdown("Silakan lengkapi parameter data pelanggan berikut untuk mengevaluasi risiko churn.")
st.divider()

# ===== Data Demografi =====
st.subheader("Data Demografi")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Usia", min_value=1, max_value=100, value=35)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    country = st.text_input("Negara", value="Indonesia")
with col2:
    city = st.text_input("Kota", value="Jakarta")
    is_premium_user = st.selectbox("Premium User", [0, 1])

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

st.write("") # Memberi sedikit jarak sebelum tombol

if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
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

    st.divider()
    st.subheader("Hasil Evaluasi")

    # Menggunakan kolom metrik untuk hasil yang lebih bersih
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_preprocessed)[0]
            st.metric(label="Probabilitas Churn", value=f"{proba[1]*100:.2f}%")
        else:
            st.metric(label="Probabilitas Churn", value="N/A")

    with res_col2:
        if prediction[0] == 1:
            st.error("Status: Risiko Tinggi (Berpotensi Churn)")
        else:
            st.success("Status: Aman (Pelanggan Aktif)")
