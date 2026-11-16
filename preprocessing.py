import re
NORMALISASI = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "tdk": "tidak",
    "udh": "sudah", "udah": "sudah", "blm": "belum", "bgt": "banget", "yg": "yang",
    "dgn": "dengan", "krn": "karena", "tp": "tapi", "jd": "jadi", "aja": "saja",
    "bs": "bisa", "sm": "sama", "klo": "kalau", "kalo": "kalau", "lg": "lagi",
    "apk": "aplikasi", "app": "aplikasi", "cs": "customer service", "brg": "barang",
    "lemot": "lambat", "lelet": "lambat", "eror": "error", "mantul": "mantap",
    "bales": "balas", "seller": "penjual", "makasih": "terima kasih",
}
STOPWORDS = set("""yang dan di ke dari ini itu dengan untuk pada juga saja lagi sih dong deh
kok nya ya kan pun lah tuh nih adalah akan atau karena jadi sudah masih ada saya kamu
kami kita dia mereka""".split())
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    STEMMER = StemmerFactory().create_stemmer()
except ImportError:
    STEMMER = None
def normalisasi(teks):
    t = str(teks).lower()
    t = re.sub(r"http\S+|www\.\S+|@\w+|#\w+", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    t = re.sub(r"[^a-z\s.,!?]", " ", t)
    kata = [NORMALISASI.get(k, k) for k in re.findall(r"[a-z]+|[.,!?]", t)]
    return " ".join(kata)
def bersihkan(teks, pakai_stemming=False):
    t = re.sub(r"[.,!?]", " ", normalisasi(teks))
    token = [k for k in t.split() if k not in STOPWORDS and len(k) > 1]
    if pakai_stemming and STEMMER:
        token = [STEMMER.stem(k) for k in token]
    return " ".join(token)
if __name__ == "__main__":
    contoh = "Apk nya LEMOT bgt!!! udh update tp gk ngaruh 😡 https://x.com"
    print("Asli       :", contoh)
    print("Normalisasi:", normalisasi(contoh))
    print("Bersih     :", bersihkan(contoh))
