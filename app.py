import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Status Gizi Balita",
    layout="centered"
)

st.title("Prediksi Status Gizi Balita")
st.write("Upload file CSV untuk melakukan prediksi status gizi balita.")

# =========================
# UPLOAD FILE
# =========================
uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Silakan upload file CSV terlebih dahulu.")
    st.stop()

# =========================
# BACA DATA
# =========================
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error("Gagal membaca file CSV")
    st.stop()

st.subheader("Preview Data")
st.dataframe(df.head())

# =========================
# VALIDASI KOLOM (PENTING)
# =========================
required_columns = [
    "Umur (bulan)",
    "Tinggi Badan (cm)",
    "Jenis Kelamin",
    "Status Gizi"
]

for col in required_columns:
    if col not in df.columns:
        st.error(f"Kolom '{col}' tidak ditemukan di CSV.")
        st.stop()

# =========================
# ENCODING
# =========================
le_jk = LabelEncoder()
le_status = LabelEncoder()

df["JK_enc"] = le_jk.fit_transform(df["Jenis Kelamin"])
df["Status_enc"] = le_status.fit_transform(df["Status Gizi"])

# =========================
# DATA SPLIT
# =========================
X = df[["Umur (bulan)", "Tinggi Badan (cm)", "JK_enc"]]
y = df["Status_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL (KNN)
# =========================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# =========================
# EVALUASI
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Hasil Evaluasi Model")
st.success(f"Akurasi Model: {accuracy:.2f}")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# =========================
# PREDIKSI MANUAL
# =========================
st.subheader("Prediksi Manual")

umur = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=12)
tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=130.0, value=75.0)
jk = st.selectbox("Jenis Kelamin", le_jk.classes_)

jk_enc = le_jk.transform([jk])[0]

if st.button("Prediksi Status Gizi"):
    hasil = model.predict([[umur, tinggi, jk_enc]])[0]
    status = le_status.inverse_transform([hasil])[0]
    st.success(f"Status Gizi Prediksi: **{status}**")
