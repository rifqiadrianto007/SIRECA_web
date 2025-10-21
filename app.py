from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import cv2
import numpy as np
from torchvision import transforms
import os
from model.model import SmoothCNN

app = FastAPI()

# Folder untuk file statis (css, uploads)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmoothCNN().to(device)
model.load_state_dict(torch.load("checkpoints/smoothcnn.pth", map_location=device))
model.eval()

# Halaman utama
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Proses smoothing
@app.post("/process", response_class=HTMLResponse)
async def process(request: Request, image: UploadFile = File(...)):
    contents = await image.read()
    filename = image.filename
    input_path = os.path.join("static/uploads", filename)
    output_path = os.path.join("static/uploads", f"result_{filename}")

    # Simpan gambar upload
    with open(input_path, "wb") as f:
        f.write(contents)

    # Proses smoothing
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t).clamp(0,1)

    out_np = (out[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, out_bgr)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_filename": filename,
        "result_filename": f"result_{filename}"
    })