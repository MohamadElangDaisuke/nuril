import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from math import pi

# =======================================================
# 1. LOAD DATASET
# =======================================================
st.set_page_config(page_title="SPK Klasifikasi Tingkat Stres", layout="wide")
st.title("🧠 Sistem Pendukung Keputusan – Klasifikasi Tingkat Stres Mahasiswa")

DATA_FILE = "nuril.csv"

try:
    df_raw = pd.read_csv(DATA_FILE, header=None)
except:
    st.error("❌ File 'nuril.csv' tidak ditemukan! Letakkan file di folder app.py")
    st.stop()

# dataset 1 kolom → split dengan ;
df = df_raw[0].str.split(";", expand=True)

# baris pertama = header
df.columns = df.iloc[0]
df = df.drop(index=0).reset_index(drop=True)

# convert ke numerik
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="ignore")

# kolom terakhir = tingkat stres (1–5)
target_col = df.columns[-1]
features = df.columns[:-1]

# =======================================================
# 2. Buat Label Klasifikasi
# =======================================================
def map_label(v):
    if v <= 2:
        return 0   # stres rendah
    elif v == 3:
        return 1   # stres sedang
    else:
        return 2   # stres tinggi

df["Label"] = df[target_col].apply(map_label)

label_map = {
    0: ("Stres Rendah", "Kondisi sehat, keluhan rendah, dan stabil.", "green"),
    1: ("Stres Sedang", "Mulai muncul tekanan akademik, butuh manajemen waktu yang baik.", "orange"),
    2: ("Stres Tinggi", "Risiko burnout tinggi, keluhan intens, perlu perhatian lebih.", "red")
}

# =======================================================
# 3. Scaling + Model Training
# =======================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
y = df["Label"]

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

st.success("✔ Model klasifikasi (Random Forest) berhasil dilatih!")

# =======================================================
# 4. PCA Visualisasi
# =======================================================
st.header("📊 Visualisasi PCA (Dataset)")

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

fig1, ax1 = plt.subplots()
scatter = ax1.scatter(pca_result[:,0], pca_result[:,1], c=y, cmap="viridis")
plt.title("PCA – Dataset")
plt.xlabel("Komponen Utama 1")
plt.ylabel("Komponen Utama 2")
st.pyplot(fig1)

# =======================================================
# 5. FORM PREDIKSI MANUAL (Bahasa Indonesia)
# =======================================================
st.header("📝 Prediksi Manual (Skala 1–5)")

# Mapping nama asli kolom → nama tampilan Indonesia
feature_labels = {
    "Kindly Rate your Sleep Quality": "Kualitas Tidur",
    "How many times a week do you suffer headaches": "Frekuensi Sakit Kepala per Minggu",
    "How would you rate you academic performance": "Performa Akademik",
    "how would you rate your study load": "Beban Belajar",
    "How many times a week you practice extracurricular activities": "Frekuensi Aktivitas Ekstrakurikuler",
    "How would you rate your stress levels": "Tingkat Stres (Asli Dataset)"
}

with st.form("manual_pred"):
    manual_input = {}

    for col in features:
        label_indonesia = feature_labels.get(col, col)
        manual_input[col] = st.slider(
            f"{label_indonesia} (1 = sangat rendah, 5 = sangat tinggi)",
            1, 5, 3
        )

    submit = st.form_submit_button("Prediksi Sekarang")


if submit:
    row = pd.DataFrame([manual_input])
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]

    label, desc, color = label_map[pred]

    # ==========================
    # OUTPUT PREDIKSI
    # ==========================
    st.success(f"🔮 Hasil Prediksi Tingkat Stres: **{label}**")
    st.write(f"🩺 **Kondisi:** {desc}")

    # ==========================
    # GAUGE PREDIKSI
    # ==========================
    st.subheader("📟 Indikator Tingkat Stres")

    fig2, ax2 = plt.subplots(figsize=(4,2))
    ax2.barh([""], [pred+1], color=color)
    ax2.set_xlim(0, 3)
    ax2.set_yticklabels([])
    ax2.set_title(f"Tingkat Stres: {label}")
    st.pyplot(fig2)

    # ==========================
    # PCA POSISI INPUT
    # ==========================
    st.subheader("📌 Posisi Input di PCA Plot")

    fig3, ax3 = plt.subplots()
    ax3.scatter(pca_result[:,0], pca_result[:,1], c=y, alpha=0.3)
    pca_input = pca.transform(row_scaled)
    ax3.scatter(pca_input[0][0], pca_input[0][1], color="red", s=200, marker="X")
    ax3.set_title("Posisi Input pada PCA")
    st.pyplot(fig3)

    # ==========================
    # Radar Chart
    # ==========================
    st.subheader("🕸 Radar Chart Perbandingan")

    cluster_means = df.groupby("Label")[features].mean()
    compare = cluster_means.loc[pred].values.tolist()
    input_vals = row.values.tolist()[0]

    labels = list(features)
    stats = compare + compare[:1]
    input_stats = input_vals + input_vals[:1]
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    fig4 = plt.figure(figsize=(5,5))
    ax4 = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], labels)
    ax4.plot(angles, stats, linewidth=1, label="Rata-rata Kelompok")
    ax4.fill(angles, stats, alpha=0.1)
    ax4.plot(angles, input_stats, linewidth=2, label="Nilai Input")
    ax4.fill(angles, input_stats, alpha=0.2)
    plt.legend()
    st.pyplot(fig4)

    # =======================================================
    # PENJELASAN KENAPA DAPAT LABEL
    # =======================================================
    st.header("🧩 Penjelasan Hasil Prediksi")
    explanation = []

    if manual_input["How many times a week do you suffer headaches"] >= 4:
        explanation.append("• Frekuensi sakit kepala cukup sering.")
    if manual_input["Kindly Rate your Sleep Quality"] <= 2:
        explanation.append("• Kualitas tidur tergolong rendah.")
    if manual_input["how would you rate your study load"] >= 4:
        explanation.append("• Beban belajar tergolong tinggi.")
    if manual_input["How many times a week you practice extracurricular activities"] <= 2:
        explanation.append("• Aktivitas fisik rendah.")
    if manual_input["How would you rate you academic performance"] <= 2:
        explanation.append("• Performa akademik relatif menurun.")

    if explanation:
        st.write("Alasan kamu mendapatkan label ini:")
        for e in explanation:
            st.write(e)
    else:
        st.write("Nilai kamu seimbang, tidak ada indikator berat yang dominan.")

    # =======================================================
    # TABEL INTERPRETASI LABEL
    # =======================================================
    st.subheader("📋 Tabel Arti Tingkat Stres")

    label_table = pd.DataFrame({
        "Label": ["Stres Rendah", "Stres Sedang", "Stres Tinggi"],
        "Keterangan": [
            "Kondisi sehat dan stabil.",
            "Mulai muncul tekanan akademik.",
            "Risiko stres berat dan burnout."
        ]
    })

    st.dataframe(label_table)
