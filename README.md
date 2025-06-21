# 🧠 Customer Churn Prediction App

Aplikasi ini memprediksi kemungkinan pelanggan melakukan churn (berhenti) berdasarkan perilaku historis mereka. Dibangun menggunakan Python dan Streamlit, serta menggunakan model machine learning `RandomForestClassifier`.

---

## 🚀 Fitur Utama

- 📂 Upload file CSV data pelanggan
- 🧠 Mode **Training**: Melatih model dari data historis (dengan kolom `churn`)
- 🔮 Mode **Prediksi**: Memprediksi kemungkinan churn dari data pelanggan baru (tanpa kolom `churn`)
- 💾 Menyimpan model hasil training ke file `.pkl`
- 📊 Tabel prediksi dengan highlight risiko tinggi
- 📈 Histogram distribusi probabilitas churn
- 🧠 Analisis **Feature Importance** (pentingnya masing-masing fitur)

---

## 🗂️ Struktur File yang Dibutuhkan

### ✅ Format data **training (dengan churn)**:

```csv
customer_id,customer_name,age,income,purchase_frequency,total_spent,churn
1,Ayu Lestari,25,50000000,5,3000000,0
2,Budi Santoso,34,70000000,2,2000000,1
...
customer_id,customer_name,age,income,purchase_frequency,total_spent
11,Rina Andriani,32,58000000,2,1800000
12,Samsul Bahri,40,72000000,4,4000000
...

pip install -r requirements.txt
streamlit run app.py

