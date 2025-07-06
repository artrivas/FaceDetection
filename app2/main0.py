#!/usr/bin/env python3
"""
API de detección de rostros y clasificación de emociones
--------------------------------------------------------
• FastAPI + Ultralytics YOLOv8/YOLOv11 + EfficientNet‑B0
• Devuelve, por cada rostro, la caja, la emoción, un emoji,
  un color HEX y la confianza.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import warnings
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

# ─────────────────────────────────────────  Ajustes globales ──

# (Opcional) silenciar el warning de torchvision
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision.models._utils",
)

YOLO_WEIGHTS     = "yolov11n-face.pt"          # Asegúrate de que exista
EMOTION_WEIGHTS  = "modelo_efficientnet.pt"    # Asegúrate de que exista
DEVICE           = "cpu"                       # Cambia a "cuda" si tienes GPU

# ───────────────────────────────────────────────  FastAPI ──

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajusta en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────  Modelos ──

# 1. Detector de rostros (YOLO)
modelo_yolo = YOLO(YOLO_WEIGHTS)

# 2. Clasificador de emociones (EfficientNet‑B0)
modelo_emociones = models.efficientnet_b0(weights=None)
modelo_emociones.classifier[1] = nn.Linear(1280, 7)

state_dict = torch.load(EMOTION_WEIGHTS, map_location=DEVICE)
modelo_emociones.load_state_dict(state_dict)
modelo_emociones.eval()

# ───────────────────────────────────────  Constantes útiles ──

EMOTIONS = {
    0: {"name": "Happy",    "emoji": "😊", "color": "#10B981"},
    1: {"name": "Neutral",  "emoji": "😐", "color": "#6B7280"},
    2: {"name": "Sad",      "emoji": "😢", "color": "#3B82F6"},
    3: {"name": "Surprise", "emoji": "😮", "color": "#F59E0B"},
    4: {"name": "Fear",     "emoji": "😨", "color": "#8B5CF6"},
    5: {"name": "Disgust",  "emoji": "🤢", "color": "#84CC16"},
    6: {"name": "Anger",    "emoji": "😠", "color": "#EF4444"},
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────  Funciones core ──

def detectar_emocion(face_img: np.ndarray) -> dict:
    """
    Recibe una ROI de rostro en BGR (OpenCV) y devuelve un dict con:
    nombre, emoji, color y confianza
    """
    try:
        tensor = transform(face_img).unsqueeze(0)  # (1, 3, 224, 224)
        with torch.no_grad():
            salida = modelo_emociones(tensor)
            probas = F.softmax(salida, dim=1)
            clase = torch.argmax(probas, dim=1).item()
            confianza = float(probas[0, clase])

        info = EMOTIONS.get(clase, {"name": "Unknown", "emoji": "❓", "color": "#9CA3AF"})
        return {
            "nombre": info["name"],
            "emoji": info["emoji"],
            "color": info["color"],
            "confianza": round(confianza, 4),
        }
    except Exception as e:
        # Log en servidor
        print("Error detectando emoción:", e)
        return {
            "nombre": "Error",
            "emoji": "❌",
            "color": "#000000",
            "confianza": 0.0,
        }

# ───────────────────────────────────────────────  Endpoints ──

@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    """
    POST /detectar
    Body: multipart/form-data con campo 'file' (imagen)
    Return: JSON con lista de resultados por rostro.
    """
    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    resultados = modelo_yolo(img)[0]   # Primer (y único) frame
    emociones = []

    for box in resultados.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Validación rápida
        if x2 <= x1 or y2 <= y1:
            continue

        rostro = img[y1:y2, x1:x2]
        if rostro.size == 0:
            continue

        emocion = detectar_emocion(rostro)

        emociones.append({
            "box": [x1, y1, x2, y2],
            "emocion": emocion["nombre"],
            "emoji": emocion["emoji"],
            "color": emocion["color"],
            "confianza": emocion["confianza"],
        })

    return JSONResponse(content={"resultados": emociones})

# ───────────────────────────────────────────  CLI / Uvicorn ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,    # Recarga en caliente para desarrollo
    )
