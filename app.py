"""
Dashboard Sistem Rekomendasi Smartphone
Content-Based Filtering + Collaborative Filtering (Matrix Factorization/SVD)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sistem Rekomendasi Smartphone",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ----------------------------------------------------------------------------
# LOAD & PREPROCESSING DATA (mengikuti alur notebook)
# ----------------------------------------------------------------------------
@st.cache_data
def load_raw_data():
    data_df = pd.read_csv(os.path.join(DATA_DIR, "cellphones_data.csv"))
    rating_df = pd.read_csv(os.path.join(DATA_DIR, "cellphones_ratings.csv"))
    user_df = pd.read_csv(os.path.join(DATA_DIR, "cellphones_users.csv"))
    return data_df, rating_df, user_df


@st.cache_data
def preprocess(data_df, rating_df, user_df):
    # Tahap 1 & 2: gabungkan rating + data ponsel + data user
    combined = rating_df.merge(data_df, on="cellphone_id", how="inner")
    final_dataset = combined.merge(user_df, on="user_id", how="inner")

    # Bersihkan missing value
    final_dataset = final_dataset.dropna().copy()

    # Buang rating anomali (nilai 18)
    final_dataset = final_dataset.loc[final_dataset["rating"] != 18].copy()

    # Standarisasi kolom occupation
    final_dataset["occupation"] = final_dataset["occupation"].str.lower()
    final_dataset["occupation"] = final_dataset["occupation"].str.replace(
        "healthare", "healthcare", regex=False
    )

    # Dataset ponsel unik (untuk content-based)
    phone_master = data_df.drop_duplicates(subset=["cellphone_id"]).reset_index(drop=True)

    return final_dataset, phone_master


# ----------------------------------------------------------------------------
# CONTENT-BASED FILTERING
# ----------------------------------------------------------------------------
@st.cache_resource
def build_content_based_model(phone_master):
    content_data = phone_master.copy()

    numeric_cols = ["performance", "main camera", "battery size", "screen size", "weight"]
    numeric_cols = [c for c in numeric_cols if c in content_data.columns]

    # Normalisasi min-max fitur numerik
    for col in numeric_cols:
        min_v, max_v = content_data[col].min(), content_data[col].max()
        if max_v > min_v:
            content_data[f"{col}_norm"] = (content_data[col] - min_v) / (max_v - min_v)
        else:
            content_data[f"{col}_norm"] = 0.5

    # Bentuk "content profile" tekstual: brand + OS + kategori binning fitur numerik
    content_data["content_profile"] = (
        content_data["brand"].astype(str) + " " + content_data["operating system"].astype(str)
    )
    for col in numeric_cols:
        norm_col = f"{col}_norm"
        try:
            cat_col = pd.qcut(content_data[norm_col], q=4, duplicates="drop")
            cat_col = cat_col.astype(str)
        except ValueError:
            cat_col = content_data[norm_col].astype(str)
        content_data["content_profile"] += " " + cat_col

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(content_data["content_profile"].fillna(""))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(content_data.index, index=content_data["cellphone_id"]).drop_duplicates()
    return content_data, cosine_sim, indices


def get_content_based_recommendations(phone_id, content_data, cosine_sim, indices, top_n=5):
    if phone_id not in indices:
        return pd.DataFrame()
    idx = indices[phone_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]  # exclude diri sendiri

    phone_idx = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    result = content_data.iloc[phone_idx].copy()
    result["similarity_score"] = scores
    return result


# ----------------------------------------------------------------------------
# COLLABORATIVE FILTERING (Matrix Factorization / SVD)
# ----------------------------------------------------------------------------
@st.cache_resource
def build_collaborative_model(final_dataset, n_components=15):
    rating_matrix = final_dataset.pivot_table(
        index="user_id", columns="cellphone_id", values="rating", aggfunc="mean"
    ).fillna(0)

    user_ids = rating_matrix.index.tolist()
    item_ids = rating_matrix.columns.tolist()

    R = rating_matrix.values
    user_means = np.true_divide(R.sum(axis=1), (R != 0).sum(axis=1), where=(R != 0).sum(axis=1) != 0)
    user_means = np.nan_to_num(user_means)
    R_demeaned = R - user_means.reshape(-1, 1) * (R != 0)

    n_comp = min(n_components, min(R.shape) - 1) if min(R.shape) > 1 else 1
    n_comp = max(n_comp, 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    latent_matrix = svd.fit_transform(R_demeaned)
    reconstructed = np.dot(latent_matrix, svd.components_) + user_means.reshape(-1, 1)

    pred_df = pd.DataFrame(reconstructed, index=user_ids, columns=item_ids)
    return pred_df, rating_matrix


def get_cf_recommendations(user_id, pred_df, rating_matrix, phone_master, top_n=5):
    if user_id not in pred_df.index:
        return pd.DataFrame()

    already_rated = rating_matrix.loc[user_id]
    already_rated_ids = already_rated[already_rated > 0].index.tolist()

    predictions = pred_df.loc[user_id].drop(labels=already_rated_ids, errors="ignore")
    predictions = predictions.sort_values(ascending=False).head(top_n)

    result = phone_master[phone_master["cellphone_id"].isin(predictions.index)].copy()
    result["predicted_rating"] = result["cellphone_id"].map(predictions.to_dict())
    result = result.sort_values("predicted_rating", ascending=False)
    result["predicted_rating"] = result["predicted_rating"].clip(1, 5)
    return result


# ----------------------------------------------------------------------------
# LOAD SEMUA DATA & MODEL
# ----------------------------------------------------------------------------
data_df, rating_df, user_df = load_raw_data()
final_dataset, phone_master = preprocess(data_df, rating_df, user_df)
content_data, cosine_sim, indices = build_content_based_model(phone_master)
pred_df, rating_matrix = build_collaborative_model(final_dataset)


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGASI
# ----------------------------------------------------------------------------
st.sidebar.title("📱 Menu Dashboard")
page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "🏠 Beranda",
        "📊 Eksplorasi Data",
        "🔍 Rekomendasi Content-Based",
        "👥 Rekomendasi Collaborative Filtering",
        "ℹ️ Tentang Proyek",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: {len(phone_master)} smartphone • {len(user_df)} pengguna • "
    f"{len(rating_df)} rating"
)

# ----------------------------------------------------------------------------
# HALAMAN: BERANDA
# ----------------------------------------------------------------------------
if page == "🏠 Beranda":
    st.title("📱 Sistem Rekomendasi Smartphone")
    st.markdown(
        "Dashboard ini mengimplementasikan sistem rekomendasi smartphone berbasis "
        "**Content-Based Filtering** (kesamaan spesifikasi) dan **Collaborative "
        "Filtering** (pola rating pengguna, via matrix factorization/SVD)."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Smartphone", len(phone_master))
    col2.metric("Jumlah Pengguna", len(user_df))
    col3.metric("Jumlah Rating", len(rating_df))
    col4.metric("Jumlah Brand", phone_master["brand"].nunique())

    st.markdown("### Cara Menggunakan")
    st.markdown(
        """
        - **📊 Eksplorasi Data**: melihat gambaran umum dataset (distribusi brand, harga, OS, dll).
        - **🔍 Rekomendasi Content-Based**: pilih satu smartphone, sistem akan menampilkan
          smartphone lain dengan spesifikasi paling mirip.
        - **👥 Rekomendasi Collaborative Filtering**: pilih user, sistem akan memprediksi
          smartphone yang mungkin disukai berdasarkan pola rating pengguna lain yang mirip.
        """
    )

    st.markdown("### Contoh Data Smartphone")
    st.dataframe(phone_master.head(10), use_container_width=True)

# ----------------------------------------------------------------------------
# HALAMAN: EKSPLORASI DATA (EDA)
# ----------------------------------------------------------------------------
elif page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data")

    tab1, tab2, tab3 = st.tabs(["Smartphone", "Rating", "Pengguna"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            brand_count = phone_master["brand"].value_counts().reset_index()
            brand_count.columns = ["brand", "jumlah"]
            fig = px.bar(brand_count, x="brand", y="jumlah", title="Distribusi Brand Smartphone")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            os_count = phone_master["operating system"].value_counts().reset_index()
            os_count.columns = ["os", "jumlah"]
            fig = px.pie(os_count, names="os", values="jumlah", title="Distribusi Sistem Operasi")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.histogram(phone_master, x="price", nbins=15, title="Distribusi Harga (USD)")
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.histogram(phone_master, x="performance", nbins=15, title="Distribusi Skor Performa")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Tabel Lengkap Data Smartphone")
        st.dataframe(phone_master, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(rating_df, x="rating", nbins=10, title="Distribusi Rating (Raw)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            merged = final_dataset.copy()
            top_rated = (
                merged.groupby(["cellphone_id", "brand", "model"])["rating"]
                .mean()
                .reset_index()
                .sort_values("rating", ascending=False)
                .head(10)
            )
            top_rated["label"] = top_rated["brand"] + " " + top_rated["model"]
            fig = px.bar(
                top_rated, x="rating", y="label", orientation="h",
                title="Top 10 Smartphone dengan Rating Rata-rata Tertinggi",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Catatan: dataset rating mentah mengandung nilai anomali (rating=18) yang "
            "sudah difilter pada proses cleaning yang dipakai model Collaborative Filtering."
        )

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(user_df, x="age", nbins=15, title="Distribusi Usia Pengguna")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            gender_count = user_df["gender"].value_counts().reset_index()
            gender_count.columns = ["gender", "jumlah"]
            fig = px.pie(gender_count, names="gender", values="jumlah", title="Distribusi Gender")
            st.plotly_chart(fig, use_container_width=True)

        occ_count = (
            user_df["occupation"].str.lower().str.replace("healthare", "healthcare", regex=False)
            .value_counts().reset_index().head(15)
        )
        occ_count.columns = ["occupation", "jumlah"]
        fig = px.bar(occ_count, x="jumlah", y="occupation", orientation="h", title="Top 15 Pekerjaan Pengguna")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# HALAMAN: CONTENT-BASED FILTERING
# ----------------------------------------------------------------------------
elif page == "🔍 Rekomendasi Content-Based":
    st.title("🔍 Rekomendasi Content-Based Filtering")
    st.markdown(
        "Merekomendasikan smartphone lain yang **spesifikasinya paling mirip** "
        "(brand, OS, performa, kamera, baterai, layar, berat) dengan smartphone yang dipilih."
    )

    phone_master["label"] = phone_master["brand"] + " - " + phone_master["model"]
    selected_label = st.selectbox("Pilih Smartphone", phone_master["label"].tolist())
    selected_row = phone_master[phone_master["label"] == selected_label].iloc[0]
    top_n = st.slider("Jumlah rekomendasi", 3, 10, 5)

    st.markdown("#### Smartphone yang Dipilih")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brand", selected_row["brand"])
    c2.metric("OS", selected_row["operating system"])
    c3.metric("Harga", f"${selected_row['price']}")
    c4.metric("Performa", selected_row["performance"])

    if st.button("Tampilkan Rekomendasi", type="primary"):
        recs = get_content_based_recommendations(
            selected_row["cellphone_id"], content_data, cosine_sim, indices, top_n=top_n
        )
        if recs.empty:
            st.warning("Tidak ditemukan rekomendasi untuk smartphone ini.")
        else:
            st.markdown(f"#### Top-{top_n} Smartphone Paling Mirip")
            display_cols = [
                "brand", "model", "operating system", "performance",
                "main camera", "battery size", "price", "similarity_score",
            ]
            display_cols = [c for c in display_cols if c in recs.columns]
            show = recs[display_cols].rename(columns={"similarity_score": "skor kemiripan"})
            show["skor kemiripan"] = show["skor kemiripan"].round(3)
            st.dataframe(show, use_container_width=True)

            fig = px.bar(
                recs, x="similarity_score", y=recs["brand"] + " " + recs["model"],
                orientation="h", title="Skor Kemiripan", labels={"y": "Smartphone"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# HALAMAN: COLLABORATIVE FILTERING
# ----------------------------------------------------------------------------
elif page == "👥 Rekomendasi Collaborative Filtering":
    st.title("👥 Rekomendasi Collaborative Filtering")
    st.markdown(
        "Memprediksi rating smartphone untuk seorang pengguna menggunakan **matrix "
        "factorization (SVD)** dari pola rating seluruh pengguna, lalu merekomendasikan "
        "smartphone dengan prediksi rating tertinggi yang belum pernah dinilai user tersebut."
    )

    available_users = sorted(pred_df.index.tolist())
    selected_user = st.selectbox("Pilih User ID", available_users)
    top_n = st.slider("Jumlah rekomendasi", 3, 10, 5, key="cf_topn")

    user_history = final_dataset[final_dataset["user_id"] == selected_user][
        ["brand", "model", "rating"]
    ].sort_values("rating", ascending=False)

    st.markdown("#### Riwayat Rating User Ini")
    if user_history.empty:
        st.info("User ini belum memiliki riwayat rating pada data bersih.")
    else:
        st.dataframe(user_history, use_container_width=True)

    if st.button("Tampilkan Rekomendasi", type="primary", key="cf_button"):
        recs = get_cf_recommendations(selected_user, pred_df, rating_matrix, phone_master, top_n=top_n)
        if recs.empty:
            st.warning("Tidak ditemukan rekomendasi baru untuk user ini (mungkin sudah menilai semua ponsel).")
        else:
            st.markdown(f"#### Top-{top_n} Rekomendasi untuk User {selected_user}")
            display_cols = ["brand", "model", "operating system", "price", "predicted_rating"]
            show = recs[display_cols].rename(columns={"predicted_rating": "prediksi rating"})
            show["prediksi rating"] = show["prediksi rating"].round(2)
            st.dataframe(show, use_container_width=True)

            fig = px.bar(
                recs, x="predicted_rating", y=recs["brand"] + " " + recs["model"],
                orientation="h", title="Prediksi Rating", labels={"y": "Smartphone"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# HALAMAN: TENTANG PROYEK
# ----------------------------------------------------------------------------
else:
    st.title("ℹ️ Tentang Proyek")
    st.markdown(
        """
        Dashboard ini adalah implementasi sederhana dari proyek **Sistem Rekomendasi
        Smartphone** menggunakan dua pendekatan:

        **1. Content-Based Filtering**
        - Menggunakan TF-IDF pada profil konten (brand, OS, dan kategori performa/kamera/
          baterai/layar/berat yang telah dinormalisasi dan di-binning).
        - Menghitung *cosine similarity* antar smartphone.
        - Cocok untuk kasus cold-start item baru dan hasilnya mudah dijelaskan.

        **2. Collaborative Filtering**
        - Membentuk matriks rating user-item, lalu menerapkan **matrix factorization
          (Truncated SVD)** untuk memprediksi rating yang belum diberikan pengguna.
        - Pendekatan ini adalah versi ringan dari model Neural Collaborative Filtering
          (embedding + dense layers) pada notebook asli, dipilih agar dashboard tetap
          ringan dan responsif tanpa perlu training ulang model deep learning setiap kali
          dijalankan.

        **Sumber Data**: [Kaggle - Cellphones Recommendation Dataset](https://www.kaggle.com/datasets/meirnizri/cellphones-recommendations/data)

        Dibangun dengan **Streamlit**, **scikit-learn**, dan **Plotly**.
        """
    )
