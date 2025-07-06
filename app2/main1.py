#!/usr/bin/env python3
"""
API de detección de rostros y clasificación de emociones ─ versión webcam/GET
-----------------------------------------------------------------------------

• FastAPI + Ultralytics YOLOv8/YOLOv11 + EfficientNet‑B0
• Cada GET /detectar captura un fotograma de la cámara 0
  y devuelve, por rostro detectado:
     - caja    [x1, y1, x2, y2]
     - track_id  (si el tracker lo asigna)
     - emoción  nombre, emoji, color, confianza
"""

from fastapi import FastAPI, HTTPException
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

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision.models._utils",
)

YOLO_WEIGHTS     = "yolov11n-face.pt"          # Asegúrate de que exista
EMOTION_WEIGHTS  = "modelo_efficientnet.pt"    # Asegúrate de que exista
DEVICE           = "cpu"                       # Cambia a "cuda" si tienes GPU
CAMERA_INDEX     = 0                           # 0 = webcam principal

# ───────────────────────────────────────────────  FastAPI ──

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajusta en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────  Modelos ──

# 1. Detector/Tracker de rostros
modelo_yolo = YOLO(YOLO_WEIGHTS)

# 2. Clasificador de emociones
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

# ─────────────────────────────────────────  Recursos globales ──

# Abrimos la webcam una sola vez y la compartimos
CAP = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not CAP.isOpened():
    raise RuntimeError(f"No se pudo abrir la cámara índice {CAMERA_INDEX}")

# ─────────────────────────────────────────  Funciones core ──

def detectar_emocion(face_img: np.ndarray) -> dict:
    """Clasifica la emoción en un rostro BGR (OpenCV)."""
    try:
        tensor = transform(face_img).unsqueeze(0)
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
        print("Error detectando emoción:", e)
        return {
            "nombre": "Error",
            "emoji": "❌",
            "color": "#000000",
            "confianza": 0.0,
        }

# ───────────────────────────────────────────────  Endpoints ──

@app.get("/detectar")
async def detectar():
    """
    GET /detectar
    Captura 1 fotograma de la webcam, aplica YOLOv* con tracking
    y devuelve las emociones detectadas.
    """
    ok, frame = CAP.read()
    if not ok or frame is None:
        raise HTTPException(status_code=500, detail="No se pudo leer de la cámara")

    # YOLO con seguimiento persistente
    resultados = modelo_yolo.track(frame, persist=True)[0]  # Primer resultado

    respuestas = []
    for box in resultados.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if x2 <= x1 or y2 <= y1:
            continue
        rostro = frame[y1:y2, x1:x2]
        if rostro.size == 0:
            continue

        emocion = detectar_emocion(rostro)

        # track_id puede no existir en la primera llamada
        track_id = int(box.id.item()) if box.id is not None else None

        respuestas.append({
            "box": [x1, y1, x2, y2],
            "track_id": track_id,
            "emocion": emocion["nombre"],
            "emoji": emocion["emoji"],
            "color": emocion["color"],
            "confianza": emocion["confianza"],
        })

    return JSONResponse(content={"resultados": respuestas})

# ───────────────────────────────────────────  Shutdown Hook ──

@app.on_event("shutdown")
def shutdown_event():
    if CAP.isOpened():
        CAP.release()

# ───────────────────────────────────────────  CLI / Uvicorn ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
