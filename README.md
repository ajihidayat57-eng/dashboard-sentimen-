# Dashboard Analisis Sentimen Berbasis Aspek – Toko Online

Unggah file CSV/Excel berisi ulasan → dashboard menampilkan aspek layanan yang paling perlu ditingkatkan.

## Isi folder
| File | Fungsi |
|---|---|
| `app.py` | Tampilan dashboard (Streamlit) |
| `aspek.py` | Deteksi aspek + skor prioritas |
| `preprocessing.py` | Pembersihan teks |
| `requirements.txt` | Daftar paket untuk Streamlit Cloud |
| `model/model_sentimen.joblib` | Model sentimen (positif/negatif) |
| `data/ulasan_berlabel.csv` | Data latih cadangan: dipakai otomatis jika model gagal dimuat |
| `data/contoh_upload.csv` | Contoh file untuk dicoba diunggah ke dashboard |

> ⚠️ Model dan kedua file di folder `data/` saat ini berasal dari **data DUMMY (buatan)**, hanya supaya dashboard
> bisa langsung dicoba. Untuk skripsi, ganti dengan milikmu (lihat bagian bawah).

## Deploy ke Streamlit Cloud
1. Buka github.com → **+ → New repository** → nama `dashboard-sentimen` → **Public** → **Create repository**.
2. Klik **uploading an existing file** → seret SEMUA isi folder ini (termasuk folder `model` dan `data`) → **Commit changes**.
3. Buka share.streamlit.io → login dengan GitHub → **Create app → Deploy a public app from GitHub**.
4. Repository: `namamu/dashboard-sentimen`, Branch: `main`, Main file path: `app.py` → **Deploy**.
5. Tunggu 3–5 menit → dashboard terbuka di `https://<nama>.streamlit.app`. Coba unggah `data/contoh_upload.csv`.

## Mengganti dengan data & model milikmu
Di GitHub, buka folder terkait → **Add file → Upload files** → unggah file dengan nama yang SAMA (menimpa yang lama):
- `model/model_sentimen.joblib` ← hasil `train_model.py` di Colab
- `data/ulasan_berlabel.csv` ← file berlabel milikmu (kolom `ulasan` dan `sentimen`)

Streamlit Cloud akan otomatis memperbarui dashboard dalam 1–2 menit.
Kalau model dari Colab gagal dimuat (beda versi scikit-learn), dashboard otomatis melatih ulang model dari
`data/ulasan_berlabel.csv`, jadi tidak error.
