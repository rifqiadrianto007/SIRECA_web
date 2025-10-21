from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import cv2
import numpy as np
from torchvision import transforms
import os
from model.model import SmoothCNN

app = FastAPI()

# Konfigurasi folder
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmoothCNN().to(device)
model.load_state_dict(torch.load("checkpoints/smoothcnn.pth", map_location=device))
model.eval()

UPLOAD_DIR = "static/uploads"
RESULT_DIR = "static/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
async def process_image(
    request: Request,
    image: UploadFile = File(...),
    level: int = Form(...)
):
    # Simpan gambar upload
    image_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(image_path, "wb") as f:
        f.write(await image.read())

    # Baca gambar
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)

    # Proses smoothing
    with torch.no_grad():
        output = model(img_t).clamp(0, 1)
    result = (output[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # Simpan hasil
    result_filename = f"result_{level}_{image.filename}"
    result_path = os.path.join(RESULT_DIR, result_filename)
    cv2.imwrite(result_path, result_bgr)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "before_path": f"/{image_path}",
        "after_path": f"/{result_path}",
        "download_path": f"/download/{result_filename}",
        "level": level
    })

@app.get("/download/{filename}")
async def download_image(filename: str):
    path = os.path.join(RESULT_DIR, filename)
    return FileResponse(path, filename=filename)