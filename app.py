# app.py — SPK: Perbandingan CBR / Klasterisasi / Klasifikasi
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

st.set_page_config(page_title="SPK Perbandingan Metode - Tingkat Stres", layout="wide")
st.title("🧠 SPK: Perbandingan CBR, Klasterisasi, dan Klasifikasi (Tingkat Stres)")

# ----------------------------
# LOAD DATA
# ----------------------------
DATA_FILE = "nuril.csv"

try:
    df_raw = pd.read_csv(DATA_FILE, header=None)
except Exception:
    st.error("❌ File 'nuril.csv' tidak ditemukan di folder yang sama dengan app.py.")
    st.stop()

# parse ; single-column CSV
df = df_raw[0].str.split(";", expand=True)
df.columns = df.iloc[0]
df = df.drop(index=0).reset_index(drop=True)

# convert numeric where possible
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# define features and target
target_col = df.columns[-1]  # last column is stress level (1-5)
features = list(df.columns[:-1])

# map label (0,1,2)
def map_label(v):
    if v <= 2:
        return 0
    elif v == 3:
        return 1
    else:
        return 2

df["Label"] = df[target_col].apply(map_label)

label_map = {
    0: ("Stres Rendah", "Kondisi sehat, keluhan rendah, stabil.", "green"),
    1: ("Stres Sedang", "Mulai muncul tekanan akademik.", "orange"),
    2: ("Stres Tinggi", "Risiko burnout; perlu perhatian.", "red")
}

# ----------------------------
# Sidebar menu
# ----------------------------
menu = st.sidebar.selectbox("Menu", [
    "Dataset",
    "Klasifikasi (Random Forest)",
    "Klasterisasi (K-Means)",
    "CBR (Case-Based Reasoning)",
    "Prediksi Manual",
    "Perbandingan Metode"
])

# ----------------------------
# Helpers: prepare scaler + X_scaled
# ----------------------------
@st.cache_data
def prepare_scaled_data(df, features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return scaler, X_scaled

scaler, X_scaled = prepare_scaled_data(df, features)
y = df["Label"].values

# ----------------------------
# GLOBAL RF MODEL (latih pada seluruh data untuk prediksi)
# ----------------------------
@st.cache_resource
def train_global_rf(X, y, n_estimators=200):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)
    return rf

# buat model global (dipakai di Prediksi Manual & CBR/KMeans mapping)
rf_global = train_global_rf(X_scaled, y, n_estimators=200)

# ----------------------------
# MENU: DATASET
# ----------------------------
if menu == "Dataset":
    st.header("📁 Data & Statistik Deskriptif")
    st.subheader("Preview dataset")
    st.dataframe(df.head())

    st.subheader("Ringkasan statistik")
    st.dataframe(df[features + [target_col]].describe())

    st.subheader("Distribusi tiap fitur")
    for col in features:
        fig, ax = plt.subplots()
        df[col].astype(int).plot.hist(ax=ax, bins=5)
        ax.set_xlabel(col)
        st.pyplot(fig)

    st.subheader("Distribusi target (Tingkat Stres asli 1–5)")
    fig, ax = plt.subplots()
    df[target_col].astype(int).value_counts().sort_index().plot.bar(ax=ax)
    ax.set_xlabel("Tingkat Stres (1-5)")
    st.pyplot(fig)

# ----------------------------
# MENU: Klasifikasi
# ----------------------------
elif menu == "Klasifikasi (Random Forest)":
    st.header("🔎 Klasifikasi — Random Forest (Evaluasi)")

    st.info("Training RandomForest dengan split train/test. Evaluasi: accuracy, confusion matrix, classification report, feature importance.")

    # train-test split
    test_size = st.sidebar.slider("Test size (%)", 10, 40, 20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100, random_state=42, stratify=y)

    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, step=50)
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    # predictions
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.subheader(f"Akurasi (test set): {acc:.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(["Rendah","Sedang","Tinggi"]); ax.set_yticklabels(["Rendah","Sedang","Tinggi"])
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j,i,val,ha="center",va="center")
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    st.subheader("Feature Importance")
    imp = pd.DataFrame({"Fitur": features, "Importance": clf.feature_importances_}).sort_values("Importance", ascending=False)
    st.dataframe(imp)
    fig2, ax2 = plt.subplots()
    ax2.barh(imp["Fitur"], imp["Importance"])
    st.pyplot(fig2)

    st.subheader("Cross-validation (Stratified K-Fold)")
    cv_folds = st.slider("Jumlah fold CV", 3, 10, 5)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")
    st.write("Akurasi per fold:", np.round(scores,4))
    st.write("Rata-rata akurasi:", np.round(scores.mean(),4))

# ----------------------------
# MENU: Klasterisasi
# ----------------------------
elif menu == "Klasterisasi (K-Means)":
    st.header("🧭 Klasterisasi — K-Means")

    clust_mode = st.radio("Mode pemilihan jumlah cluster", ("Paksa k=3 (default)", "Gunakan Elbow method & pilih k"))

    if clust_mode.startswith("Paksa"):
        k = 3
    else:
        st.write("Elbow: hitung SSE untuk k dari 1 sampai max_k, lalu pilih k")
        max_k = st.sidebar.slider("Max k untuk Elbow", 5, 15, 10)
        sse = []
        K_range = range(1, max_k+1)
        for ktest in K_range:
            kmeans_test = KMeans(n_clusters=ktest, random_state=42, n_init=10)
            kmeans_test.fit(X_scaled)
            sse.append(kmeans_test.inertia_)
        fig_elb, ax_elb = plt.subplots()
        ax_elb.plot(K_range, sse, '-o')
        ax_elb.set_xlabel("k")
        ax_elb.set_ylabel("SSE (inertia)")
        ax_elb.set_title("Elbow Method")
        st.pyplot(fig_elb)

        k = st.slider("Pilih k berdasarkan Elbow plot", 2, max_k, 3)

    # run kmeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_clusters = df.copy()
    df_clusters["Cluster"] = clusters

    st.subheader(f"Hasil K-Means (k={k}) — Jumlah tiap cluster")
    st.write(df_clusters["Cluster"].value_counts().sort_index())

    # silhouette (if k>1)
    if k > 1:
        sil = silhouette_score(X_scaled, clusters)
        st.write(f"Silhouette Score: {sil:.4f}")

    st.subheader("PCA plot (Cluster)")
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_scaled)
    figc, axc = plt.subplots()
    scatter = axc.scatter(pca_res[:,0], pca_res[:,1], c=clusters, cmap="tab10", alpha=0.7)
    st.pyplot(figc)

    st.subheader("Perbandingan Cluster vs Label Asli (cross-tab)")
    cros = pd.crosstab(df_clusters["Cluster"], df_clusters["Label"])
    st.dataframe(cros)

    st.subheader("Rata-rata fitur per cluster")
    st.dataframe(df_clusters.groupby("Cluster")[features].mean())

# ----------------------------
# MENU: CBR
# ----------------------------
elif menu == "CBR (Case-Based Reasoning)":
    st.header("🧠 Case-Based Reasoning (CBR) — Prediksi berdasarkan kasus serupa")

    st.markdown("CBR: cari kasus paling mirip (top-k) pada dataset, gunakan mayoritas label kasus mirip sebagai prediksi.")

    # pilih metric
    metric = st.selectbox("Pilih metric similarity / distance:", ("euclidean (jarak)", "cosine (kesamaan)"))

    k_sim = st.slider("Ambil top-k kasus paling mirip:", 1, 10, 5)

    st.write("Isi nilai input untuk mencari kasus serupa:")
    # show sliders to input values (use original feature names)
    input_case = {}
    for col in features:
        input_case[col] = st.slider(col, 1, 5, 3, key=f"cbr_{col}")

    if st.button("Cari Kasus Mirip (CBR)"):
        # prepare array
        row_scaled = scaler.transform(pd.DataFrame([input_case]))  # scaler from earlier

        if metric.startswith("euclidean"):
            dists = pairwise_distances(X_scaled, row_scaled, metric="euclidean").flatten()
            order = np.argsort(dists)
            st.write("Metric: Euclidean (jarak). Jarak lebih kecil = lebih mirip.")
            top_idx = order[:k_sim]
            dist_values = dists[top_idx]
        else:
            # cosine similarity (higher better)
            cos = cosine_similarity(X_scaled, row_scaled).flatten()
            order = np.argsort(-cos)  # descending
            st.write("Metric: Cosine (kesamaan). Nilai lebih tinggi = lebih mirip.")
            top_idx = order[:k_sim]
            dist_values = 1 - cos[top_idx]

        results = df.iloc[top_idx].copy()
        results["Distance"] = np.round(dist_values, 4)
        results["Label"] = results["Label"].map(lambda r: label_map[r][0])
        st.subheader("Top-k kasus paling mirip")
        st.dataframe(results[features + ["Label", "Distance"]].reset_index(drop=True))

        # majority vote
        voted = df.iloc[top_idx]["Label"].mode()
        if len(voted) > 0:
            pred_label = voted.iloc[0]
            st.success(f"Prediksi CBR (mayoritas top-{k_sim}): {label_map[pred_label][0]}")
        else:
            st.write("Tidak ada mayoritas (tie). Menampilkan label-label top-k:")
            st.write(df.iloc[top_idx]["Label"].map(lambda r: label_map[r][0]).values)

# ----------------------------
# MENU: Prediksi Manual
# ----------------------------
elif menu == "Prediksi Manual":
    st.header("📝 Prediksi Manual — Bandingkan hasil metode")

    st.write("Isi input (slider), lalu aplikasi akan menampilkan prediksi dari:")
    st.write("- Random Forest (klasifikasi) — model terlatih")
    st.write("- CBR (top-k) — prediksi berdasar kasus mirip")
    st.write("- K-Means cluster (lihat cluster terdekat)")

    # sliders in Indonesian labels
    feature_labels = {
        "Kindly Rate your Sleep Quality": "Kualitas Tidur",
        "How many times a week do you suffer headaches": "Frekuensi Sakit Kepala",
        "How would you rate you academic performance": "Performa Akademik",
        "how would you rate your study load": "Beban Belajar",
        "How many times a week you practice extracurricular activities": "Frekuensi Ekstrakurikuler",
        "How would you rate your stress levels": "Tingkat Stres (Asli Dataset)"
    }

    manual_input = {}
    for col in features:
        label_id = feature_labels.get(col, col)
        manual_input[col] = st.slider(f"{label_id} (1–5)", 1, 5, 3, key=f"pred_{col}")

    if st.button("Prediksi & Bandingkan"):
        row = pd.DataFrame([manual_input])
        row_scaled = scaler.transform(row)

        # RF prediction using global model (trained on full dataset)
        rf_pred = rf_global.predict(row_scaled)[0]
        rf_proba = rf_global.predict_proba(row_scaled)[0]
        st.subheader("Hasil Random Forest (model global)")
        st.write(f"Label: **{label_map[rf_pred][0]}** — {label_map[rf_pred][1]}")
        prob_df = pd.DataFrame({"Label": [label_map[i][0] for i in range(3)], "Probabilitas": np.round(rf_proba,3)})
        st.table(prob_df)

        # CBR prediction: default euclidean, top-5
        dists = pairwise_distances(X_scaled, row_scaled, metric="euclidean").flatten()
        top_idx = np.argsort(dists)[:5]
        cbr_votes = df.iloc[top_idx]["Label"].mode()
        cbr_pred = cbr_votes.iloc[0] if len(cbr_votes)>0 else df.iloc[top_idx]["Label"].iloc[0]
        st.subheader("Hasil CBR (top-5, Euclidean)")
        st.write(f"Prediksi CBR (mayoritas): **{label_map[cbr_pred][0]}**")
        st.dataframe(df.iloc[top_idx][features + ["Label"]].reset_index(drop=True).assign(Distance=np.round(dists[top_idx],4)))

        # KMeans nearest centroid (k=3)
        # KMeans nearest centroid
        kmeans_for_pred = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
        centroid_dists = pairwise_distances(kmeans_for_pred.cluster_centers_, row_scaled, metric="euclidean").flatten()
        nearest_cluster = np.argmin(centroid_dists)

        st.subheader("Hasil K-Means (cluster terdekat centroid)")
        st.write(f"Cluster terdekat: {nearest_cluster}")

        # map cluster -> majority label in that cluster
        cluster_assign = kmeans_for_pred.predict(X_scaled)
        df_tmp = df.copy()
        df_tmp["Cluster"] = cluster_assign

        maj_label = df_tmp[df_tmp["Cluster"]==nearest_cluster]["Label"].mode().iloc[0]
        st.write(f"Label mayoritas pada cluster ini: **{label_map[maj_label][0]}**")

        # ======= Tambahkan Tabel Anggota Cluster =======
        st.write("### 🔍 Contoh Data Dalam Cluster Ini")
        cluster_members = df_tmp[df_tmp["Cluster"] == nearest_cluster]
        st.dataframe(cluster_members.head(10))  # tampilkan 10 baris awal

        # ======= Tambahkan Tabel Nilai Centroid =======
        st.write("### 📌 Nilai Centroid Cluster")
        centroid_values = pd.DataFrame(
            [kmeans_for_pred.cluster_centers_[nearest_cluster]],
            columns=features
        )
        st.dataframe(centroid_values)

# ----------------------------
# MENU: Perbandingan Metode
# ----------------------------
elif menu == "Perbandingan Metode":
    st.header("📋 Perbandingan Ringkas: CBR vs K-Means vs RandomForest")

    # run RF CV
    rf = RandomForestClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores = cross_val_score(rf, X_scaled, y, cv=skf, scoring="accuracy")
    rf_mean = rf_scores.mean()

    # kmeans with k=3 evaluation by matching to labels (purity)
    kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
    clusters3 = kmeans3.labels_
    df_tmp = df.copy(); df_tmp["Cluster3"] = clusters3
    # purity: sum of max intersection per cluster / N
    purity = 0
    for c in np.unique(clusters3):
        cluster_labels = df_tmp[df_tmp["Cluster3"]==c]["Label"]
        if len(cluster_labels) > 0:
            most_common = cluster_labels.value_counts().max()
            purity += most_common
    purity = purity / len(df_tmp)

    # quick CBR estimate: using leave-one-out accuracy by top-5 majority
    preds_cbr = []
    for i in range(len(X_scaled)):
        row_i = X_scaled[i].reshape(1,-1)
        dists = pairwise_distances(np.delete(X_scaled, i, axis=0), row_i, metric="euclidean").flatten()
        idxs = np.argsort(dists)[:5]
        labels_pool = np.delete(df["Label"].values, i)[idxs]
        if len(labels_pool) > 0:
            preds_cbr.append(pd.Series(labels_pool).mode().iloc[0])
        else:
            preds_cbr.append(df["Label"].iloc[i])
    cbr_acc = (np.array(preds_cbr) == df["Label"].values).mean()

    comp_df = pd.DataFrame({
        "Metode": ["Random Forest (CV5)", "K-Means (k=3) Purity", "CBR (LOO top-5 approx)"],
        "Tipe": ["Supervised", "Unsupervised", "Case-based (lazy)"],
        "Metrik": [f"Akurasi mean {rf_mean:.4f}", f"Purity {purity:.4f}", f"Akurasi approx {cbr_acc:.4f}"],
        "Kelebihan": [
            "Akurasi tinggi, dapat probabilitas",
            "Menemukan pola tanpa label",
            "Mencerminkan keputusan berbasis kasus nyata"
        ],
        "Kekurangan": [
            "Butuh data berlabel, rawan overfit jika data sedikit",
            "Tidak menghasilkan label, perlu interpretasi",
            "Bergantung kualitas & representasi kasus"
        ]
    })

    st.dataframe(comp_df)

    st.markdown("**Catatan:** Purity untuk K-Means dihitung dengan membandingkan cluster -> label asli; bukan 'akurasi' sejati karena klasterisasi adalah unsupervised. CBR dihitung sekadar pendekatan leave-one-out untuk memberi gambaran performa.")

# EOF
