from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np
import cv2

app = FastAPI()

# Mengizinkan akses dari browser lokal
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/smooth")
async def smooth_image(file: UploadFile, intensity: int = Form(5)):
    """
    Endpoint smoothing gambar dengan Gaussian blur
    tanpa menyimpan file ke sistem.
    """
    # Baca gambar sebagai array numpy
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Tentukan ukuran kernel berdasarkan slider (1–20)
    ksize = max(1, int(intensity))
    if ksize % 2 == 0:  # kernel harus ganjil
        ksize += 1

    # Proses smoothing dengan Gaussian Blur
    smoothed = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Encode hasil ke format JPEG (tanpa menyimpan file)
    _, encoded_img = cv2.imencode(".jpg", smoothed)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")