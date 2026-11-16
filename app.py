import io
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from aspek import analisis, skor_prioritas, kata_teratas, ASPEK

st.set_page_config(page_title="Dashboard Sentimen Ulasan", layout="wide")
WARNA = {"negatif": "#D64545", "positif": "#2E9E6B"}
WARNA_PRIORITAS = {"Tinggi": "#D64545", "Sedang": "#E5A23B", "Rendah": "#2E9E6B"}
IKON = {"Tinggi": "🔴", "Sedang": "🟠", "Rendah": "🟢"}


# ======================================================================
# MODEL
# ======================================================================
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


def tebak_kolom(kolom, kandidat):
    for k in kolom:
        if str(k).lower().strip() in kandidat:
            return k
    return None


def gambar_wordcloud(frekuensi, warna):
    """Word cloud. Jika paket wordcloud tidak tersedia, tampilkan grafik batang."""
    if not frekuensi:
        st.info("Belum ada kata untuk ditampilkan.")
        return
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap="Reds" if warna == "negatif" else "Greens",
                       prefer_horizontal=0.9).generate_from_frequencies(frekuensi)
        st.image(wc.to_array(), use_container_width=True)
    except ImportError:
        d = pd.DataFrame(list(frekuensi.items()), columns=["kata", "frekuensi"])
        fig = px.bar(d.sort_values("frekuensi"), x="frekuensi", y="kata", orientation="h",
                     color_discrete_sequence=[WARNA[warna]])
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# SIDEBAR: UNGGAH & PENGATURAN
# ======================================================================
st.sidebar.title("Analisis Ulasan Toko Online")
berkas = st.sidebar.file_uploader("Unggah file ulasan (CSV/Excel)", type=["csv", "xlsx"])

if berkas is None:
    st.title("Dashboard Analisis Sentimen Berbasis Aspek")
    st.write("Unggah file berisi kolom ulasan pelanggan di sidebar. Sistem akan menunjukkan "
             "aspek layanan yang paling perlu ditingkatkan.")
    st.write("**Aspek yang dianalisis:** " + ", ".join(ASPEK.keys()))
    st.stop()

df = baca_file(berkas)
kolom_list = list(df.columns)
tebak = tebak_kolom(kolom_list, {"ulasan", "review", "content", "komentar", "teks"})
kolom = st.sidebar.selectbox("Kolom ulasan", kolom_list,
                             index=kolom_list.index(tebak) if tebak else 0)

opsi_tgl = ["(tidak ada)"] + kolom_list
tebak_tgl = tebak_kolom(kolom_list, {"tanggal", "date", "at", "waktu"})
kolom_tgl = st.sidebar.selectbox("Kolom tanggal (untuk tren & filter)", opsi_tgl,
                                 index=opsi_tgl.index(tebak_tgl) if tebak_tgl else 0)
ada_tanggal = kolom_tgl != "(tidak ada)"

# ---------- Filter tanggal ----------
if ada_tanggal:
    df["_tanggal"] = pd.to_datetime(df[kolom_tgl], errors="coerce").dt.tz_localize(None)
    if df["_tanggal"].notna().any():
        awal, akhir = df["_tanggal"].min().date(), df["_tanggal"].max().date()
        rentang = st.sidebar.date_input("Filter periode", value=(awal, akhir),
                                        min_value=awal, max_value=akhir)
        if isinstance(rentang, (tuple, list)) and len(rentang) == 2:
            mulai, selesai = pd.Timestamp(rentang[0]), pd.Timestamp(rentang[1]) + pd.Timedelta(days=1)
            df = df[(df["_tanggal"] >= mulai) & (df["_tanggal"] < selesai)]
    else:
        st.sidebar.warning("Kolom tanggal tidak bisa dibaca sebagai tanggal.")
        ada_tanggal = False

if df.empty:
    st.warning("Tidak ada ulasan pada periode yang dipilih.")
    st.stop()

# ======================================================================
# ANALISIS
# ======================================================================
with st.spinner("Menganalisis ulasan..."):
    per_ulasan, per_aspek = proses(df.reset_index(drop=True), kolom)
if ada_tanggal:
    per_aspek["tanggal"] = per_aspek["id"].map(per_ulasan.set_index("id")["_tanggal"])
prioritas = skor_prioritas(per_aspek)
total = len(per_ulasan)

# ---------- Ringkasan ----------
st.title("Hasil Analisis Ulasan")
if ada_tanggal:
    st.caption(f"Periode: {per_ulasan['_tanggal'].min():%d %b %Y} – {per_ulasan['_tanggal'].max():%d %b %Y}")
jumlah = per_ulasan["sentimen"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total ulasan", total)
c2.metric("Positif", f"{jumlah.get('positif', 0) / total:.0%}")
c3.metric("Negatif", f"{jumlah.get('negatif', 0) / total:.0%}")
if len(prioritas):
    c4.metric("Prioritas #1", prioritas.loc[0, "aspek"])

tab1, tab2, tab3, tab4 = st.tabs(["Prioritas & Rekomendasi", "Detail Aspek", "Tren", "Data & Unduh"])

# ======================================================================
# TAB 1: PRIORITAS & REKOMENDASI
# ======================================================================
with tab1:
    if prioritas.empty:
        st.info("Tidak ada aspek spesifik yang terdeteksi pada ulasan ini.")
    else:
        kiri, kanan = st.columns([3, 2])
        with kiri:
            st.subheader("Aspek yang perlu ditingkatkan")
            fig = px.bar(prioritas.sort_values("skor"), x="skor", y="aspek", orientation="h",
                         color="prioritas", color_discrete_map=WARNA_PRIORITAS, text="skor",
                         labels={"skor": "Skor prioritas (0-100)", "aspek": ""})
            st.plotly_chart(fig, use_container_width=True)
        with kanan:
            st.subheader("Rekomendasi")
            for _, r in prioritas.head(3).iterrows():
                keluhan = per_aspek[(per_aspek["aspek"] == r["aspek"]) &
                                    (per_aspek["sentimen_aspek"] == "negatif")]["klausa"]
                kata = ", ".join(list(kata_teratas(keluhan, 5).keys())) or "-"
                contoh = keluhan.iloc[0] if len(keluhan) else "-"
                st.markdown(
                    f"{IKON[r['prioritas']]} **{r['aspek']}** — prioritas {r['prioritas'].lower()}  \n"
                    f"{r['persen_negatif']:.0f}% dari {int(r['total'])} ulasan tentang aspek ini bernada negatif.  \n"
                    f"Keluhan dominan: *{kata}*  \n"
                    f"Contoh: \"{contoh}\""
                )
        st.caption("Skor = % negatif x (jumlah sebutan aspek / sebutan terbanyak). "
                   "Tinggi: >= 50% negatif, Sedang: 30-49%, Rendah: < 30%.")
        st.dataframe(prioritas, hide_index=True, use_container_width=True)

# ======================================================================
# TAB 2: DETAIL ASPEK + WORD CLOUD
# ======================================================================
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

    st.divider()
    a1, a2 = st.columns(2)
    pilih = a1.selectbox("Pilih aspek", sorted(per_aspek["aspek"].unique()))
    sentimen = a2.radio("Sentimen", ["negatif", "positif"], horizontal=True)
    contoh = per_aspek[(per_aspek["aspek"] == pilih) & (per_aspek["sentimen_aspek"] == sentimen)]

    st.subheader(f"Word cloud: {pilih} ({sentimen})")
    gambar_wordcloud(kata_teratas(contoh["klausa"], 40), sentimen)

    st.write(f"**{len(contoh)} ulasan**")
    st.dataframe(contoh[[kolom, "klausa"]].rename(columns={kolom: "ulasan asli",
                                                           "klausa": "bagian yang membahas aspek"}),
                 hide_index=True, use_container_width=True)

# ======================================================================
# TAB 3: TREN
# ======================================================================
with tab3:
    if not ada_tanggal:
        st.info("Pilih kolom tanggal di sidebar untuk melihat tren.")
    else:
        satuan = st.radio("Kelompokkan per", ["Minggu", "Bulan"], horizontal=True)
        d = per_aspek[(per_aspek["aspek"] != "Umum") & per_aspek["tanggal"].notna()].copy()
        d["periode"] = d["tanggal"].dt.to_period("W" if satuan == "Minggu" else "M").dt.start_time
        daftar_aspek = sorted(d["aspek"].unique())
        bawaan = [a for a in prioritas["aspek"].head(3) if a in daftar_aspek]
        dipilih = st.multiselect("Aspek yang ditampilkan", daftar_aspek, default=bawaan)

        g = (d[d["aspek"].isin(dipilih)]
             .groupby(["periode", "aspek"])
             .agg(total=("sentimen_aspek", "size"),
                  negatif=("sentimen_aspek", lambda s: (s == "negatif").sum()))
             .reset_index())
        g["% negatif"] = (g["negatif"] / g["total"] * 100).round(1)

        st.subheader("Persentase ulasan negatif per aspek")
        fig = px.line(g, x="periode", y="% negatif", color="aspek", markers=True,
                      hover_data={"total": True}, labels={"periode": ""})
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Garis yang turun berarti keluhan pada aspek itu berkurang, artinya perbaikan berhasil.")

        st.subheader("Jumlah ulasan per periode")
        u = per_ulasan.dropna(subset=["_tanggal"]).copy()
        u["periode"] = u["_tanggal"].dt.to_period("W" if satuan == "Minggu" else "M").dt.start_time
        v = u.groupby(["periode", "sentimen"]).size().reset_index(name="jumlah")
        fig = px.bar(v, x="periode", y="jumlah", color="sentimen", color_discrete_map=WARNA,
                     labels={"periode": ""})
        st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# TAB 4: DATA & UNDUH
# ======================================================================
with tab4:
    tampil = per_ulasan.drop(columns=["_tanggal"], errors="ignore")
    st.dataframe(tampil[[kolom, "sentimen"]], hide_index=True, use_container_width=True)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        prioritas.to_excel(writer, sheet_name="Prioritas", index=False)
        tampil.to_excel(writer, sheet_name="Per Ulasan", index=False)
        per_aspek.to_excel(writer, sheet_name="Per Aspek", index=False)
    st.download_button("Unduh laporan (Excel)", buffer.getvalue(), "laporan_ulasan.xlsx")
