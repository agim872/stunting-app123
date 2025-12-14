import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(
    page_title="Prediksi Status Gizi Balita",
    layout="wide"
)

st.title("📊 Aplikasi Prediksi Status Gizi Balita")
st.write("Upload dataset CSV untuk melakukan analisis dan pemodelan.")

# =============================
# Upload File
# =============================
uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

    # =============================
    # Encoding
    # =============================
    le_gender = LabelEncoder()
    le_status = LabelEncoder()

    df["JK_enc"] = le_gender.fit_transform(df["Jenis Kelamin"])
    df["Status_enc"] = le_status.fit_transform(df["Status Gizi"])

    # =============================
    # Visualisasi
    # =============================
    st.subheader("📈 Visualisasi Data")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(df["Umur (bulan)"], bins=30)
        ax.set_title("Distribusi Umur Balita")
        ax.set_xlabel("Umur (bulan)")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.hist(df["Tinggi Badan (cm)"], bins=30)
        ax.set_title("Distribusi Tinggi Badan")
        ax.set_xlabel("Tinggi (cm)")
        ax.set_ylabel("Jumlah")
        st.pyplot(fig)

    st.subheader("Distribusi Status Gizi")
    fig, ax = plt.subplots()
    df["Status Gizi"].value_counts().plot(kind="bar", ax=ax)
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

    # =============================
    # Modeling
    # =============================
    X = df[["Umur (bulan)", "Tinggi Badan (cm)", "JK_enc"]]
    y = df["Status_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- KNN ----
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, pred_knn)

    # ---- Random Forest ----
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, pred_rf)

    st.subheader("📊 Hasil Evaluasi Model")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Akurasi KNN", f"{acc_knn:.3f}")
        st.text("Classification Report KNN")
        st.text(classification_report(y_test, pred_knn))

    with col2:
        st.metric("Akurasi Random Forest", f"{acc_rf:.3f}")
        st.text("Classification Report RF")
        st.text(classification_report(y_test, pred_rf))

    # =============================
    # Feature Importance
    # =============================
    importances = rf.feature_importances_
    feature_names = X.columns

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.subheader("⭐ Feature Importance")
    st.dataframe(fi_df)

    fig, ax = plt.subplots()
    sns.barplot(
        x="Importance",
        y="Feature",
        data=fi_df,
        ax=ax
    )
    ax.set_title("Pentingnya Fitur (Random Forest)")
    st.pyplot(fig)

    st.success(
        f"Fitur paling berpengaruh: {fi_df.iloc[0,0]}"
    )

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
