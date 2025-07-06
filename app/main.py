from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import asyncio
import json
import base64
from typing import Dict, Any
import logging
import tensorflow as tf
from ultralytics import YOLO


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion Recognition API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diccionario de emociones
EMOTIONS = {
    1: {"name": "Neutral", "emoji": "😐", "color": "#6B7280"},
    0: {"name": "Happy", "emoji": "😊", "color": "#10B981"},
    2: {"name": "Sad", "emoji": "😢", "color": "#3B82F6"},
    5: {"name": "Surprise", "emoji": "😮", "color": "#F59E0B"},
    6: {"name": "Fear", "emoji": "😨", "color": "#8B5CF6"},
    4: {"name": "Disgust", "emoji": "🤢", "color": "#84CC16"},
    3: {"name": "Anger", "emoji": "😠", "color": "#EF4444"},}

class EmotionRecognizer:
    def __init__(self, model_path: str = "emotion_model_mobilenet.h5"):
        self.model_path = model_path
        self.model = None
        self.yolo = YOLO("/home/andrewkc/Projects/FaceDetection/app/yolov11n-face.pt")  # 👈 Modelo de detección de rostro
        self.yolo.to('cuda' if torch.cuda.is_available() else 'cpu')  # 👈 Enviar a GPU si existe
        self.load_model()

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Modelo de emociones cargado desde {self.model_path}")
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            self.model = None

    def preprocess_face(self, face_img):
        face_resized = cv2.resize(face_img, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)
        return face_expanded

    def predict_emotion(self, face_img):
        try:
            if self.model is None:
                return np.random.rand(8)
            face_tensor = self.preprocess_face(face_img)
            predictions = self.model.predict(face_tensor)
            return predictions[0]
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return np.random.rand(8)

    def detect_faces_and_emotions(self, frame):
        results = self.yolo(frame, verbose=False)[0]  # 👈 salida de YOLOv8
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        scores = results.boxes.conf.cpu().numpy() if results.boxes else []

        detections = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = float(scores[i])
            if conf < 0.5:  # 👈 umbral opcional
                continue

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            emotion_probs = self.predict_emotion(face_roi)
            emotion_idx = np.argmax(emotion_probs)
            confidence = emotion_probs[emotion_idx]

            detections.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "emotion": EMOTIONS[emotion_idx],
                "confidence": float(confidence),
                "all_emotions": {
                    EMOTIONS[i]["name"]: float(prob)
                    for i, prob in enumerate(emotion_probs)
                }
            })

        return detections
    
# Instancia global del reconocedor de emociones
emotion_recognizer = EmotionRecognizer()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Cliente {client_id} conectado")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Cliente {client_id} desconectado")
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps(message))

manager = ConnectionManager()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/emotions")
async def get_emotions():
    """Endpoint para obtener la lista de emociones disponibles"""
    return {"emotions": EMOTIONS}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    cap = cv2.VideoCapture(2) # 1 or 2 (in linux) for Irium camera
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        logger.error("No se pudo abrir la cámara Irium")
        await manager.send_personal_message(
            {"type": "error", "message": "No se pudo acceder a la cámara Irium"}, 
            client_id
        )
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("No se pudo leer frame de la cámara")
                continue
            
            # Detectar emociones
            results = emotion_recognizer.detect_faces_and_emotions(frame)
            
            # Codificar frame para enviar al frontend
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Enviar datos al frontend
            await manager.send_personal_message({
                "type": "frame_data",
                "frame": frame_data,
                "emotions": results,
                "timestamp": asyncio.get_event_loop().time()
            }, client_id)
            
            # Pequeña pausa para controlar FPS
            await asyncio.sleep(1/30)  # 30 FPS
            
    except WebSocketDisconnect:
        logger.info(f"Cliente {client_id} desconectado")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
    finally:
        cap.release()
        manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")