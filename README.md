# RAG API — Pedoman Kampus

Ini adalah proyek backend API berbasis **FastAPI** yang mengimplementasikan arsitektur **Retrieval-Augmented Generation (RAG)**. API ini dirancang untuk menjawab pertanyaan seputar dokumen kampus secara otomatis menggunakan teknologi pemrosesan bahasa alami (NLP).

Proyek ini sangat hemat biaya karena memadukan pemrosesan dokumen/vektor secara lokal dengan LLM gratis dari Google.

##  Teknologi yang Digunakan
- **Framework API:** FastAPI & Uvicorn
- **Orkestrasi AI:** LangChain
- **LLM Utama:** Google Gemini (`gemini-2.5-flash` via API)
- **Embedding Vektor:** HuggingFace (`all-MiniLM-L6-v2` berjalan secara lokal)
- **Database Vektor:** ChromaDB

---

##  Prasyarat
- Python 3.10 atau lebih baru.
- Akun [Google AI Studio](https://aistudio.google.com/) untuk mendapatkan Google API Key secara gratis.
- Dokumen PDF kampus (contoh: `BUKU PEDOMAN TUGAS AKHIR DAN MAGANG INTERNAL FAKULTAS INFORMATIKA.pdf`) diletakkan di root direktori proyek.

---

### Langkah Instalasi & Penggunaan

### 1. Clone Repositori & Persiapan
Clone Repositori ini dan masuk ke foldernya:
```bash
git clone <url-repo-anda>
cd portofolio
```

Disarankan membuat *virtual environment*:
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Instalasi Dependensi
Instal semua pustaka yang dibutuhkan menggunakan `pip`:
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Environment Variable
Salin file `.env.example` menjadi `.env` dan masukkan API Key Google Anda.
```bash
# Windows (Command Prompt/PowerShell):
copy .env.example .env

```
Buka file `.env` dan isi `GOOGLE_API_KEY`:
```env
GOOGLE_API_KEY=KODE_API_KEY_ANDA_DI_SINI
```

### 4. Proses Ekstraksi Vektor (Ingestion)
Sebelum server bisa menjawab pertanyaan, ubah teks dari file PDF ke dalam database vektor (ChromaDB). Jalankan script berikut **hanya sekali** (atau setiap kali file PDF berubah):
```bash
python ingest.py
```
*Script ini akan memecah dokumen PDF dan mengubahnya menjadi vektor menggunakan model lokal HuggingFace yang didownload otomatis ke komputer/server.*

### 5. Jalankan Server API
Jalankan server *development* menggunakan `uvicorn`:
```bash
uvicorn main:app --reload
```
Server akan berjalan secara default di: `http://localhost:8000`

---

##  Dokumentasi API & Endpoint

FastAPI secara otomatis menyediakan antarmuka dokumentasi Swagger UI. Setelah server berjalan, buka browser dan akses:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

### Endpoint Utama

- **`GET /`** : Health check untuk memverifikasi server berjalan normal.
- **`GET /stats`** : Melihat statistik vektor database lokal yang ada di ChromaDB.
- **`POST /chat`** : Endpoint utama RAG untuk melakukan tanya-jawab.

#### Contoh Request ke `/chat`
```json
// POST http://localhost:8000/chat
{
  "pertanyaan": "Apa syarat daftar tugas akhir?"
}
```

#### Contoh Response dari `/chat`
```json
{
  "jawaban": "Syarat mendaftar tugas akhir meliputi lulus minimal 120 SKS, IPK minimal 2.0, dan tidak ada nilai E.",
  "sumber_dokumen": [
    "BUKU PEDOMAN TUGAS AKHIR...pdf — Halaman 12",
    "BUKU PEDOMAN TUGAS AKHIR...pdf — Halaman 15"
  ]
}
```

---

## Lisensi / Penggunaan
Proyek portofolio ini dapat digunakan dan dimodifikasi secara bebas untuk kebutuhan pembelajaran maupun pengembangan lanjutan.
