# 🐔 SIRECA - Sistem Deteksi dan Smoothing Citra Ayam

Proyek ini adalah aplikasi web sederhana berbasis **Flask + Jinja2 + PyTorch** untuk menampilkan proses *smoothing* pada citra ayam.  
User dapat mengunggah gambar, menekan tombol **Proses**, lalu melihat hasil sebelum dan sesudah *smoothing* secara langsung di halaman web.

---

## ⚙️ 1. Persyaratan Sistem

Pastikan kamu sudah menginstal:

- **Python 3.10+**
- **pip** (biasanya sudah ada bersama Python)
- **virtualenv** *(opsional tapi disarankan)*

---

## 🧩 2. Instalasi Library

Langkah pertama, buka terminal di folder proyek (misalnya `SIRECA_`), lalu buat dan aktifkan *virtual environment* (opsional tapi direkomendasikan):

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan (Windows)
venv\Scripts\activate

# Aktifkan (Mac/Linux)
source venv/bin/activate
```

Kemudian install semua library yang dibutuhkan:

```bash
Pilih salah satu :
    # pip install fastapi uvicorn jinja2 torch torchvision numpy opencv-python python-multipart
    # pip install -r requirement.txt
```

Jika kamu menggunakan model berbasis GPU, pastikan PyTorch yang terinstal sesuai dengan versi CUDA kamu:  
👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## 🧠 3. Struktur Folder

Struktur proyek direkomendasikan seperti ini:

```
SIRECA_/
│
├── app.py
├── model.py
├── checkpoints/
│   └── smoothcnn.pth
│
├── static/
│   ├── uploads/
│   └── results/
│
├── templates/
│   └── index.html
│
└── README.md
```

---

## 🚀 4. Cara Menjalankan Proyek

Jalankan server Flask dengan perintah:

```bash
python app.py
```

Jika menggunakan **Uvicorn (FastAPI style)**:

```bash
python -m uvicorn app:app --reload
```

Setelah server aktif, buka browser dan akses:

👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)  
atau jika pakai Uvicorn  
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🧪 5. Fitur Utama

- Upload gambar ayam (format: `.jpg`, `.png`, `.jpeg`)
- Proses *smoothing* otomatis dengan model CNN (`smoothcnn.pth`)
- Tampilkan hasil **Before** dan **After** smoothing di halaman yang sama
- Tampilan web dibuat menggunakan **Jinja2 Template Engine**

---
