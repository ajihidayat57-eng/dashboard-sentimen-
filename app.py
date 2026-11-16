import io
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from aspek import analisis, skor_prioritas, ASPEK

st.set_page_config(page_title="Dashboard Sentimen Ulasan", layout="wide")
WARNA = {"negatif": "#D64545", "positif": "#2E9E6B"}
WARNA_PRIORITAS = {"Tinggi": "#D64545", "Sedang": "#E5A23B", "Rendah": "#2E9E6B"}


@st.cache_resource
def muat_model():
    """Pakai model terlatih. Kalau tidak ada / beda versi, latih ulang dari data berlabel."""
    try:
        return joblib.load("model/model_sentimen.joblib")
    except Exception:
        return latih_ulang()


def latih_ulang():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from preprocessing import bersihkan

    df = pd.read_csv("data/ulasan_berlabel.csv").dropna(subset=["ulasan", "sentimen"])
    df["sentimen"] = df["sentimen"].str.lower().str.strip()
    df = df[df["sentimen"].isin(["negatif", "positif"])]
    df["bersih"] = df["ulasan"].apply(bersihkan)
    df = df[df["bersih"].str.split().str.len() >= 3].drop_duplicates(subset="bersih")
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                      ("clf", MultinomialNB())])
    model.fit(df["bersih"], df["sentimen"])
    return model


@st.cache_data
def proses(df, kolom):
    return analisis(df, kolom, muat_model())


def baca_file(berkas):
    if berkas.name.endswith(".csv"):
        return pd.read_csv(berkas)
    return pd.read_excel(berkas)


# ---------- SIDEBAR ----------
st.sidebar.title("Analisis Ulasan Toko Online")
berkas = st.sidebar.file_uploader("Unggah file ulasan (CSV/Excel)", type=["csv", "xlsx"])

if berkas is None:
    st.title("Dashboard Analisis Sentimen Berbasis Aspek")
    st.write("Unggah file berisi kolom ulasan pelanggan di sidebar. Sistem akan menunjukkan "
             "aspek layanan yang paling perlu ditingkatkan.")
    st.write("**Aspek yang dianalisis:** " + ", ".join(ASPEK.keys()))
    st.stop()

df = baca_file(berkas)
kolom = st.sidebar.selectbox("Pilih kolom ulasan", df.columns,
                             index=list(df.columns).index("ulasan") if "ulasan" in df.columns else 0)

with st.spinner("Menganalisis ulasan..."):
    per_ulasan, per_aspek = proses(df, kolom)
prioritas = skor_prioritas(per_aspek)
total = len(per_ulasan)

# ---------- RINGKASAN ----------
st.title("Hasil Analisis Ulasan")
jumlah = per_ulasan["sentimen"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total ulasan", total)
c2.metric("Positif", f"{jumlah.get('positif', 0) / total:.0%}")
c3.metric("Negatif", f"{jumlah.get('negatif', 0) / total:.0%}")
if len(prioritas):
    c4.metric("Prioritas #1", prioritas.loc[0, "aspek"])

tab1, tab2, tab3 = st.tabs(["Prioritas Perbaikan", "Detail Aspek", "Data & Unduh"])

# ---------- TAB 1: PRIORITAS ----------
with tab1:
    st.subheader("Aspek yang perlu ditingkatkan")
    fig = px.bar(prioritas.sort_values("skor"), x="skor", y="aspek", orientation="h",
                 color="prioritas", color_discrete_map=WARNA_PRIORITAS, text="skor",
                 labels={"skor": "Skor prioritas (0-100)", "aspek": ""})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Skor = % negatif x (jumlah sebutan aspek / sebutan terbanyak). "
               "Tinggi: >= 50% negatif, Sedang: 30-49%, Rendah: < 30%.")
    st.dataframe(prioritas, hide_index=True, use_container_width=True)

# ---------- TAB 2: DETAIL ----------
with tab2:
    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader("Sentimen per aspek")
        d = per_aspek.groupby(["aspek", "sentimen_aspek"]).size().reset_index(name="jumlah")
        fig = px.bar(d, x="jumlah", y="aspek", color="sentimen_aspek", orientation="h",
                     color_discrete_map=WARNA, labels={"aspek": "", "sentimen_aspek": "Sentimen"})
        st.plotly_chart(fig, use_container_width=True)
    with kanan:
        st.subheader("Sentimen keseluruhan")
        fig = px.pie(per_ulasan, names="sentimen", color="sentimen",
                     color_discrete_map=WARNA, hole=0.5)
        st.plotly_chart(fig, use_container_width=True)

    pilih = st.selectbox("Lihat ulasan untuk aspek", sorted(per_aspek["aspek"].unique()))
    sentimen = st.radio("Sentimen", ["negatif", "positif"], horizontal=True)
    contoh = per_aspek[(per_aspek["aspek"] == pilih) & (per_aspek["sentimen_aspek"] == sentimen)]
    st.write(f"{len(contoh)} ulasan")
    st.dataframe(contoh[[kolom, "klausa"]], hide_index=True, use_container_width=True)

# ---------- TAB 3: DATA & UNDUH ----------
with tab3:
    st.dataframe(per_ulasan[[kolom, "sentimen"]], hide_index=True, use_container_width=True)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        prioritas.to_excel(writer, sheet_name="Prioritas", index=False)
        per_ulasan.to_excel(writer, sheet_name="Per Ulasan", index=False)
        per_aspek.to_excel(writer, sheet_name="Per Aspek", index=False)
    st.download_button("Unduh laporan (Excel)", buffer.getvalue(), "laporan_ulasan.xlsx")
