import re
import numpy as np
import pandas as pd
from preprocessing import normalisasi, bersihkan

# Kamus aspek: nama aspek -> kata kunci. Sesuaikan dengan isi ulasan di datamu.
ASPEK = {
    "Pengiriman": ["kirim", "pengiriman", "kurir", "ekspedisi", "paket", "resi", "ongkir",
                   "cod", "nyampe", "belum sampai", "lama sampai", "telat"],
    "Kualitas Produk": ["barang", "produk", "kualitas", "original", "palsu", "rusak", "cacat",
                        "sesuai", "ukuran", "bahan", "warna", "deskripsi"],
    "Harga & Promo": ["harga", "murah", "mahal", "promo", "diskon", "voucher", "cashback",
                      "gratis ongkir", "flash sale", "biaya"],
    "Pembayaran": ["bayar", "pembayaran", "transfer", "saldo", "paylater", "checkout", "top up"],
    "Pelayanan / CS": ["customer service", "pelayanan", "layanan", "respon", "admin", "penjual",
                       "chat", "balas", "ramah", "komplain"],
    "Pengembalian & Refund": ["refund", "retur", "pengembalian", "dikembalikan", "garansi"],
    "Performa Aplikasi": ["aplikasi", "error", "lambat", "loading", "crash", "bug", "update",
                          "berat", "hang", "fitur", "tampilan", "iklan"],
    "Akun & Keamanan": ["akun", "login", "otp", "verifikasi", "password", "diblokir", "keamanan"],
}

# Pemecah klausa: satu ulasan bisa berisi beberapa opini
PEMISAH = r"[.!?,]+|\b(?:tapi|tetapi|namun|cuma|sayangnya|padahal)\b"
URUTAN = {"negatif": 0, "positif": 1}


def deteksi_aspek(klausa):
    """Kembalikan daftar aspek yang kata kuncinya muncul di klausa."""
    return [aspek for aspek, kata_kunci in ASPEK.items()
            if any(re.search(rf"\b{re.escape(k)}", klausa) for k in kata_kunci)]


def analisis(df, kolom_ulasan, model):
    """Klasifikasi sentimen per ulasan dan per aspek."""
    df = df.dropna(subset=[kolom_ulasan]).reset_index(drop=True)
    df["id"] = df.index + 1
    df["sentimen"] = model.predict(df[kolom_ulasan].apply(bersihkan))

    # 1. Pecah tiap ulasan menjadi klausa, lalu cari aspeknya
    baris = []
    for _, r in df.iterrows():
        teks = normalisasi(r[kolom_ulasan])
        ketemu = False
        for klausa in re.split(PEMISAH, teks):
            klausa = klausa.strip()
            for a in deteksi_aspek(klausa):
                baris.append({"id": r["id"], "aspek": a, "klausa": klausa})
                ketemu = True
        if not ketemu:
            baris.append({"id": r["id"], "aspek": "Umum", "klausa": teks})
    hasil = pd.DataFrame(baris)

    # 2. Prediksi sentimen tiap klausa (positif / negatif)
    hasil["sentimen_aspek"] = model.predict(hasil["klausa"].apply(bersihkan))
    hasil = hasil.merge(df[["id", "sentimen", kolom_ulasan]], on="id")

    # 3. Satu aspek disebut dua kali dalam satu ulasan -> ambil yang paling negatif
    hasil["_urut"] = hasil["sentimen_aspek"].map(URUTAN)
    hasil = hasil.sort_values("_urut").drop_duplicates(["id", "aspek"]).drop(columns="_urut")
    return df, hasil.sort_values("id")


def skor_prioritas(hasil):
    """Skor = % negatif x (jumlah sebutan / sebutan terbanyak). Makin tinggi, makin prioritas."""
    d = hasil[hasil["aspek"] != "Umum"]
    t = pd.crosstab(d["aspek"], d["sentimen_aspek"])
    t = t.reindex(columns=["negatif", "positif"], fill_value=0)
    t["total"] = t.sum(axis=1)
    t["persen_negatif"] = (t["negatif"] / t["total"] * 100).round(1)
    t["skor"] = (t["persen_negatif"] * t["total"] / t["total"].max()).round(1)
    t["prioritas"] = np.select([t["persen_negatif"] >= 50, t["persen_negatif"] >= 30],
                               ["Tinggi", "Sedang"], default="Rendah")
    return t.sort_values("skor", ascending=False).reset_index()


if __name__ == "__main__":
    import joblib
    model = joblib.load("model/model_sentimen.joblib")
    contoh = pd.DataFrame({"ulasan": [
        "Pengiriman cepat tapi aplikasi sering error pas checkout",
        "CS gk bales, refund udh 2 minggu blm masuk",
        "Barang bagus bgt, harga murah",
    ]})
    per_ulasan, per_aspek = analisis(contoh, "ulasan", model)
    print(per_aspek[["id", "aspek", "sentimen_aspek", "klausa"]])
    print(skor_prioritas(per_aspek))
