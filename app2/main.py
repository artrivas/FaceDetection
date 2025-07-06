from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

# Iniciar la app
app = FastAPI()

# CORS para frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Cargar modelo YOLOv8 para rostros
modelo_yolo = YOLO("yolov11n-face.pt")

# Reconstruir arquitectura EfficientNet
modelo_emociones = models.efficientnet_b0(pretrained=False)
modelo_emociones.classifier[1] = nn.Linear(in_features=1280, out_features=7)

# Cargar pesos del modelo de emociones
state_dict = torch.load("modelo_efficientnet.pt", map_location=torch.device("cpu"))
modelo_emociones.load_state_dict(state_dict)
modelo_emociones.eval()

# Etiquetas y colores de emociones
EMOTIONS = {
    0: {"name": "Happy",    "emoji": "😊", "color": "#10B981"},
    1: {"name": "Neutral",  "emoji": "😐", "color": "#6B7280"},
    2: {"name": "Sad",      "emoji": "😢", "color": "#3B82F6"},
    3: {"name": "Surprise", "emoji": "😮", "color": "#F59E0B"},
    4: {"name": "Fear",     "emoji": "😨", "color": "#8B5CF6"},
    5: {"name": "Disgust",  "emoji": "🤢", "color": "#84CC16"},
    6: {"name": "Anger",    "emoji": "😠", "color": "#EF4444"},
}

# Preprocesamiento para el modelo de emociones
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Clasificación de emoción por rostro
def detectar_emocion(face_img):
    try:
        tensor = transform(face_img).unsqueeze(0)
        with torch.no_grad():
            salida = modelo_emociones(tensor)
            probas = F.softmax(salida, dim=1)
            clase = torch.argmax(probas, dim=1).item()
            confianza = probas[0][clase].item()

            emocion_info = EMOTIONS.get(clase, {"name": "Unknown", "emoji": "❓", "color": "#9CA3AF"})
            return {
                "nombre": emocion_info["name"],
                "emoji": emocion_info["emoji"],
                "color": emocion_info["color"],
                "confianza": round(confianza, 4)
            }
    except Exception as e:
        print("Error detectando emoción:", e)
        return {
            "nombre": "Error",
            "emoji": "❌",
            "color": "#000000",
            "confianza": 0.0
        }

# Ruta principal de detección
@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    results = modelo_yolo(img)[0]
    emociones = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
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
            "confianza": emocion["confianza"]
        })

    return JSONResponse(content={"resultados": emociones})
