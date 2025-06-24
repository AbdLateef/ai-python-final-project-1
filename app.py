import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
            padding: 2rem;
        }
        header, footer, .css-18ni7ap { visibility: hidden; }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        .stButton>button {
            background-color: #3b82f6;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1.5rem;
            font-weight: bold;
            font-size: 1rem;
            border: none;
        }
        .stDownloadButton>button {
            background-color: #10b981;
            color: white;
            font-weight: bold;
            border-radius: 6px;
            padding: 0.5rem 1.5rem;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)


# Title
st.markdown("<h1 style='text-align: center;'>🧠 Customer Churn Prediction</h1>", unsafe_allow_html=True)
MODEL_PATH = "churn_model.pkl"

# Sidebar upload
st.sidebar.markdown("## 📂 Upload Customer Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("### 📄 Data Preview")
    st.dataframe(df.head())

    has_label = 'churn' in df.columns
    features = ['age', 'income', 'purchase_frequency', 'total_spent']
    available_features = [col for col in features if col in df.columns]

    if len(available_features) < 2:
        st.error("❌ Data harus punya minimal 2 dari kolom: age, income, purchase_frequency, total_spent")
    else:
        # Scale data
        X = df[available_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train or predict
        if has_label:
            st.success("🧪 Mode: Training — Model dilatih dari data historis.")

            y = df['churn']
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, MODEL_PATH)

            y_pred = model.predict(X_test)
            st.markdown("### 🧪 Model Performance")
            st.code(classification_report(y_test, y_pred), language='text')

            churn_prob = model.predict_proba(X_scaled)[:, list(model.classes_).index(1)]
            df['churn_probability'] = churn_prob

        else:
            st.info("🔮 Mode: Prediksi — Menggunakan model tersimpan.")

            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                if 1 in model.classes_:
                    churn_prob = model.predict_proba(X_scaled)[:, list(model.classes_).index(1)]
                else:
                    churn_prob = np.zeros(X_scaled.shape[0])
                df['churn_probability'] = churn_prob
            else:
                st.error("❌ Model belum tersedia. Upload data historis terlebih dahulu.")
                st.stop()

        # Display prediction
        st.markdown("### 📋 Hasil Prediksi Churn")

        def highlight_churn(val):
            if pd.isna(val):
                return ''
            if val >= 0.8:
                return 'background-color: #ef4444; color: white;'  # strong red
            elif val >= 0.6:
                return 'background-color: #f87171; color: black;'  # medium red
            elif val >= 0.4:
                return 'background-color: #facc15; color: black;'  # yellow
            elif val >= 0.2:
                return 'background-color: #a3e635; color: black;'  # lime
            else:
                return 'background-color: #4ade80; color: black;'  # green

        display_cols = ['customer_id'] + available_features + ['churn_probability']
        if 'customer_name' in df.columns:
            display_cols.insert(1, 'customer_name')

        st.dataframe(
            df[display_cols]
            .sort_values(by='churn_probability', ascending=False)
            .style.applymap(highlight_churn, subset=['churn_probability'])
        )

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil Prediksi", csv, "prediksi_churn.csv", "text/csv")

        # Layout: side-by-side visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Distribusi Probabilitas Churn")
            fig, ax = plt.subplots()
            ax.hist(df['churn_probability'], bins=10, color='skyblue', edgecolor='black')
            ax.set_title("Distribusi Probabilitas Churn")
            ax.set_xlabel("Probabilitas")
            ax.set_ylabel("Jumlah Pelanggan")
            st.pyplot(fig)

        with col2:

            st.markdown("### 🧠 Kontribusi Fitur")
            importances = pd.Series(model.feature_importances_, index=available_features)
            importances = importances.sort_values(ascending=True)
            fig2, ax2 = plt.subplots()
            importances.plot(kind='barh', color='teal', ax=ax2)
            ax2.set_title("Kontribusi Fitur terhadap Prediksi Churn")
            st.pyplot(fig2)

            st.markdown("### 📌 Ringkasan Prediksi")

            total_customers = len(df)
            rata_rata_churn = df['churn_probability'].mean()
            high_risk = df[df['churn_probability'] >= 0.8]
            medium_risk = df[(df['churn_probability'] >= 0.6) & (df['churn_probability'] < 0.8)]
            low_risk = df[df['churn_probability'] < 0.4]

            st.markdown(f"""
            - **Total pelanggan dianalisis**: {total_customers}
            - **Rata-rata probabilitas churn**: {rata_rata_churn:.2f}
            - **Pelanggan risiko tinggi (≥ 0.8)**: {len(high_risk)}
            - **Pelanggan risiko sedang (0.6 – 0.79)**: {len(medium_risk)}
            - **Pelanggan risiko rendah (< 0.4)**: {len(low_risk)}
            """)

            # Contoh pelanggan berisiko tinggi
            if len(high_risk) > 0:
                top_risk_names = high_risk['customer_name'].head(3).tolist() if 'customer_name' in high_risk.columns else high_risk['customer_id'].head(3).tolist()
                st.markdown(f"""
                ⚠️ **Perhatian!** Terdapat **{len(high_risk)} pelanggan** dengan risiko churn sangat tinggi.
                
                Beberapa pelanggan yang perlu segera diintervensi:
                - {', '.join(map(str, top_risk_names))}
                """)
            else:
                st.success("✅ Tidak ada pelanggan dengan risiko churn sangat tinggi berdasarkan data yang diunggah.")

            # Rekomendasi otomatis
            st.markdown("### 💡 Rekomendasi Tindakan")

            if len(high_risk) > 0:
                st.markdown("""
                - 📞 **Hubungi pelanggan risiko tinggi** secara personal untuk mengetahui penyebab ketidakpuasan.
                - 🎁 Berikan **insentif loyalitas**, seperti diskon atau akses eksklusif.
                - 🤝 Tawarkan **program retensi** seperti membership, reminder berlangganan, atau konten bernilai tambah.
                """)
            elif len(medium_risk) > 0:
                st.markdown("""
                - 🧩 Pantau pelanggan risiko sedang dan lakukan kampanye re-engagement (email marketing, survey kepuasan).
                """)
            else:
                st.markdown("""
                - ✅ Mayoritas pelanggan berada dalam kondisi aman. Fokus pada **peningkatan layanan & mempertahankan kepuasan**.
                """)

else:
    st.info("⬆️ Upload file CSV terlebih dahulu untuk memulai.")
