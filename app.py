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

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("🧠 Customer Churn Prediction")

MODEL_PATH = "churn_model.pkl"

st.sidebar.header("📂 Upload Customer Data (CSV)")
uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Data Preview", df.head())

    has_label = 'churn' in df.columns
    features = ['age', 'income', 'purchase_frequency', 'total_spent']
    available_features = [col for col in features if col in df.columns]

    if len(available_features) < 2:
        st.error("❌ Data harus punya minimal 2 dari kolom: age, income, purchase_frequency, total_spent")
    else:
        X = df[available_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if has_label:
            st.success("🧠 Mode: Training — Model berhasil dilatih dari data historis.")

            y = df['churn']
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            joblib.dump(model, MODEL_PATH)

            y_pred = model.predict(X_test)
            st.write("### 📊 Model Performance")
            st.text(classification_report(y_test, y_pred))

            churn_prob = model.predict_proba(X_scaled)[:, list(model.classes_).index(1)]
            df['churn_probability'] = churn_prob

        else:
            st.info("🔮 Mode: Prediksi — Menggunakan model yang telah disimpan.")

            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                if 1 in model.classes_:
                    churn_prob = model.predict_proba(X_scaled)[:, list(model.classes_).index(1)]
                else:
                    churn_prob = np.zeros(X_scaled.shape[0])
                df['churn_probability'] = churn_prob
            else:
                st.error("❌ Model belum dilatih. Silakan upload data historis terlebih dahulu.")
                st.stop()

        st.write("### 📋 Hasil Prediksi Churn")

        def highlight_churn(val):
            return 'background-color: #ffcccc' if val >= 0.7 else ''

        display_cols = ['customer_id'] + available_features + ['churn_probability']
        if 'customer_name' in df.columns:
            display_cols.insert(1, 'customer_name')

        st.dataframe(
            df[display_cols]
            .sort_values(by='churn_probability', ascending=False)
            .style.applymap(highlight_churn, subset=['churn_probability'])
        )

        st.write("### 📈 Distribusi Probabilitas Churn")
        fig, ax = plt.subplots()
        ax.hist(df['churn_probability'], bins=10, color='skyblue', edgecolor='black')
        ax.set_title("Distribusi Probabilitas Churn")
        ax.set_xlabel("Probabilitas")
        ax.set_ylabel("Jumlah Pelanggan")
        st.pyplot(fig)

        st.write("### 🧠 Pentingnya Fitur (Feature Importance)")
        importances = pd.Series(model.feature_importances_, index=available_features)
        importances = importances.sort_values(ascending=True)
        fig2, ax2 = plt.subplots()
        importances.plot(kind='barh', color='teal', ax=ax2)
        ax2.set_title("Kontribusi Fitur terhadap Prediksi Churn")
        st.pyplot(fig2)

else:
    st.info("⬆️ Silakan unggah file CSV dengan data pelanggan untuk memulai.")
