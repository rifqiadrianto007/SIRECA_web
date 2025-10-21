from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import io
from PIL import Image
import base64

app = FastAPI()

# Folder templates (untuk HTML)
templates = Jinja2Templates(directory="templates")

# Folder static (kalau ada asset css/js)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ROUTE UTAMA
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ROUTE UNTUK PROSES SMOOTHING
@app.post("/process", response_class=HTMLResponse)
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    smoothing_level: int = Form(5)
):
    # Baca file gambar
    image_data = await file.read()
    img_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Proses smoothing (Gaussian blur)
    kernel_size = max(1, (smoothing_level // 2) * 2 + 1)
    smoothed = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Konversi gambar sebelum & sesudah ke base64 agar bisa ditampilkan langsung di HTML
    _, buffer_before = cv2.imencode('.png', img)
    _, buffer_after = cv2.imencode('.png', smoothed)
    img_base64_before = base64.b64encode(buffer_before).decode('utf-8')
    img_base64_after = base64.b64encode(buffer_after).decode('utf-8')

    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_before": img_base64_before,
        "image_after": img_base64_after,
        "smoothing_level": smoothing_level
    })