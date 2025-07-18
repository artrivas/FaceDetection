from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
import asyncio
import base64
import os
import json
from typing import List, Dict
import cv2
import numpy as np
from PIL import Image
import io
import uvicorn
import time
import logging

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import shufflenet_v2_x1_0
import timm

# YOLO import (mantener para detección de rostros)
from ultralytics import YOLO

"""
    PYTORCH MODEL - PART
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir la arquitectura del modelo híbrido (debe ser idéntica al entrenamiento)
class HybridFERModel(nn.Module):
    def __init__(self, num_classes=7):
        super(HybridFERModel, self).__init__()
        
        # ShuffleNet V2 backbone
        self.shufflenet = shufflenet_v2_x1_0(pretrained=True)
        shuffle_features = self.shufflenet.fc.in_features
        self.shufflenet.fc = nn.Identity()
        
        # EfficientViT-M2 backbone
        self.efficientvit = timm.create_model('efficientvit_m2', pretrained=True, num_classes=0)
        vit_features = self.efficientvit.num_features
        self.efficientvit.classifier = nn.Identity()
        
        # Adaptadores de características
        self.shuffle_adapter = nn.Sequential(
            nn.Linear(shuffle_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        self.vit_adapter = nn.Sequential(
            nn.Linear(vit_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        # Módulo de atención cruzada
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Fusión avanzada
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Extraer características
        shuffle_feat = self.shufflenet(x)
        vit_feat = self.efficientvit(x)
        
        # Adaptar características
        shuffle_adapted = self.shuffle_adapter(shuffle_feat)
        vit_adapted = self.vit_adapter(vit_feat)
        
        # Atención cruzada
        shuffle_attended, _ = self.cross_attention(
            shuffle_adapted.unsqueeze(1), 
            vit_adapted.unsqueeze(1), 
            vit_adapted.unsqueeze(1)
        )
        vit_attended, _ = self.cross_attention(
            vit_adapted.unsqueeze(1), 
            shuffle_adapted.unsqueeze(1), 
            shuffle_adapted.unsqueeze(1)
        )
        
        # Fusionar características
        fused = torch.cat([
            shuffle_attended.squeeze(1), 
            vit_attended.squeeze(1)
        ], dim=1)
        
        # Procesamiento final
        fused = self.fusion(fused)
        output = self.classifier(fused)
        
        return output

# Actualizar diccionario de emociones para coincidir con tu modelo
EMOTIONS = {
    0: {"name": "Anger", "emoji": "A", "color": "#FF0000"},      # Rojo fuerte
    1: {"name": "Contempt", "emoji": "C", "color": "#7C00FF"},   # Violeta intenso
    2: {"name": "Disgust", "emoji": "D", "color": "#00FF00"},    # Verde neón
    3: {"name": "Fear", "emoji": "F", "color": "#FFA500"},       # Naranja fuerte
    4: {"name": "Happiness", "emoji": "H", "color": "#FFFF00"},  # Amarillo brillante
    5: {"name": "Neutral", "emoji": "N", "color": "#FFFFFF"},    # Blanco puro
    6: {"name": "Sadness", "emoji": "SA", "color": "#00BFFF"},   # Celeste brillante
    7: {"name": "Surprise", "emoji": "SU", "color": "#FF69B4"}   # Rosado fuerte
}


class PyTorchEmotionRecognizer:
    def __init__(self, model_path: str = "best_fer_model_v8.pth", yolo_path: str = "yolov11n-face.pt"):
        self.model_path = model_path
        self.yolo_path = yolo_path
        self.model = None
        self.yolo = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Transformaciones para el modelo PyTorch
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.load_models()

    def load_models(self):
        try:
            # Cargar modelo PyTorch de emociones
            self.model = HybridFERModel(num_classes=7)
            
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Modelo PyTorch cargado desde {self.model_path}")
                logger.info(f"Mejor accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
            else:
                logger.warning(f"No se encontró el modelo en {self.model_path}, usando modelo aleatorio")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Cargar YOLO para detección de rostros
            self.yolo = YOLO(self.yolo_path)
            self.yolo.to('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Modelo YOLO cargado desde {self.yolo_path}")
            
        except Exception as e:
            logger.error(f"Error al cargar modelos: {e}")
            self.model = None
            self.yolo = None

    def preprocess_face(self, face_img):
        """Preprocesar cara para el modelo PyTorch"""
        try:
            # Convertir de BGR a RGB
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Convertir a PIL Image
            pil_image = Image.fromarray(face_rgb)
            
            # Aplicar transformaciones
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            return input_tensor
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            return None

    def predict_emotion(self, face_img):
        """Predecir emoción usando el modelo PyTorch"""
        try:
            if self.model is None:
                # Retornar probabilidades aleatorias si no hay modelo
                return np.random.rand(7)
            
            # Preprocesar imagen
            input_tensor = self.preprocess_face(face_img)
            if input_tensor is None:
                return np.random.rand(7)
            
            # Realizar predicción
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                probs_np = probabilities.cpu().numpy()[0]
            
            return probs_np
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return np.random.rand(7)

    def detect_faces_and_emotions(self, frame):
        """Detectar rostros y clasificar emociones"""
        if self.yolo is None:
            return []
        
        try:
            results = self.yolo(frame, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
            scores = results.boxes.conf.cpu().numpy() if results.boxes else []

            detections = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                conf = float(scores[i])
                
                if conf < 0.5:  # Umbral de confianza
                    continue

                # Extraer ROI del rostro
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue

                # Predecir emoción
                emotion_probs = self.predict_emotion(face_roi)
                emotion_idx = np.argmax(emotion_probs)
                confidence = emotion_probs[emotion_idx]

                detections.append({
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "label": EMOTIONS[emotion_idx]["name"],
                    "emoji": EMOTIONS[emotion_idx]["emoji"],
                    "color": EMOTIONS[emotion_idx]["color"],
                    "score": float(confidence),
                    "all_emotions": {
                        EMOTIONS[i]["name"]: float(prob)
                        for i, prob in enumerate(emotion_probs)
                    }
                })

            return detections
            
        except Exception as e:
            logger.error(f"Error en detección: {e}")
            return []

# Instancia global del reconocedor con el nuevo modelo PyTorch
emotion_recognizer = PyTorchEmotionRecognizer()

"""
    WEBSOCKETS PART - (Mantienes todo tu código WebSocket igual)
"""
app = FastAPI(title="Servidor de Imágenes WebSocket con PyTorch")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_counter = 0

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        self.connection_counter += 1
        client_id = f"client_{self.connection_counter}"
        self.active_connections[client_id] = websocket
        print(f"Cliente {client_id} conectado. Total conexiones: {len(self.active_connections)}")
        
        # Enviar ID del cliente
        await self.send_to_client(client_id, {
            "type": "connection_established",
            "clientId": client_id,
            "message": f"Conectado como {client_id}"
        })
        
        # Notificar a otros clientes
        await self.broadcast_except(client_id, {
            "type": "client_joined",
            "clientId": client_id,
            "totalClients": len(self.active_connections)
        })
        
        return client_id

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"Cliente {client_id} desconectado. Total conexiones: {len(self.active_connections)}")

    async def send_to_client(self, client_id: str, message: dict):
        """Envía mensaje a un cliente específico"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
                return True
            except Exception as e:
                print(f"Error enviando a {client_id}: {e}")
                self.disconnect(client_id)
                return False
        return False

    async def broadcast_to_all(self, message: dict, exclude_client: str = None):
        """Envía mensaje a todos los clientes"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            if exclude_client and client_id == exclude_client:
                continue
                
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                print(f"Error enviando a {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Limpiar conexiones rotas
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    async def broadcast_except(self, exclude_client: str, message: dict):
        """Envía mensaje a todos excepto al cliente especificado"""
        await self.broadcast_to_all(message, exclude_client)

    async def send_image_to_all(self, image_data: str, image_type: str = "image/jpeg", sender_id: str = "server"):
        """Envía imagen a todos los clientes conectados"""
        message = {
            "type": "image",
            "imageData": image_data,
            "imageType": image_type,
            "senderId": sender_id,
            "timestamp": int(time.time() * 1000)
        }
        await self.broadcast_to_all(message)

    async def send_image_to_client(self, client_id: str, image_data: str, image_type: str = "image/jpeg", sender_id: str = "server"):
        """Envía imagen a un cliente específico"""
        message = {
            "type": "image",
            "imageData": image_data,
            "imageType": image_type,
            "senderId": sender_id,
            "timestamp": int(time.time() * 1000)
        }
        await self.send_to_client(client_id, message)

manager = ConnectionManager()

@app.get("/")
async def get_home():
    """Página de prueba para el cliente WebSocket"""
    with open("./home.html", "r", encoding="utf-8") as file:
        html = file.read()
    return HTMLResponse(content=html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await handle_received_message(websocket, client_id, data)
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        # Notificar a otros clientes
        await manager.broadcast_to_all({
            "type": "client_left",
            "clientId": client_id,
            "totalClients": len(manager.active_connections)
        })
    except Exception as e:
        print(f"Error en WebSocket {client_id}: {e}")
        manager.disconnect(client_id)

def image_to_base64(image_path: str) -> str:
    """Convierte una imagen a base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def numpy_to_base64(image_array: np.ndarray) -> str:
    """Convierte un array numpy a base64"""
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(image_array)
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    return base64.b64encode(img_buffer.read()).decode('utf-8')

async def handle_received_message(websocket: WebSocket, client_id: str, data: str):
    """Maneja los mensajes recibidos del cliente"""
    try:
        message = json.loads(data)
        message_type = message.get("type")
        
        if message_type == "send_image":
            await handle_send_image(client_id, message)
        elif message_type == "generate_test_image":
            await handle_generate_test_image(client_id, message)
        elif message_type == "server_test_image":
            await handle_server_test_image(client_id)
        elif message_type == "start_continuous":
            await handle_start_continuous(client_id)
        elif message_type == "stop_continuous":
            await handle_stop_continuous(client_id)
        elif message_type == "ping":
            await manager.send_to_client(client_id, {"type": "pong"})
        elif message_type == "request_status":
            await handle_status_request(client_id)
        else:
            print(f"Tipo de mensaje no reconocido de {client_id}: {message_type}")
            await manager.send_to_client(client_id, {
                "type": "error",
                "message": f"Tipo de mensaje no reconocido: {message_type}"
            })
            
    except json.JSONDecodeError:
        await manager.send_to_client(client_id, {
            "type": "error",
            "message": "Formato de mensaje inválido (no es JSON válido)"
        })
    except Exception as e:
        print(f"Error procesando mensaje de {client_id}: {e}")
        await manager.send_to_client(client_id, {
            "type": "error", 
            "message": str(e)
        })

async def handle_send_image(client_id: str, message: dict):
    """Maneja el envío de imagen desde cliente con análisis de emociones PyTorch"""
    try:
        image_data = message.get("imageData")
        image_type = message.get("imageType", "image/jpeg")
        broadcast = message.get("broadcast", True)

        if not image_data:
            await manager.send_to_client(client_id, {
                "type": "error",
                "message": "No se encontró imageData en el mensaje"
            })
            return
            
        # Decodificar imagen
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is not None:
            # Detectar emociones usando el modelo PyTorch
            detections = emotion_recognizer.detect_faces_and_emotions(frame)
            
            # Dibujar detecciones en la imagen
            for det in detections:
                x, y, w, h = det["x"], det["y"], det["w"], det["h"]
                color_hex = det["color"]
                # Convertir color hex a BGR para OpenCV
                color = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))  # BGR order
                
                label = f'{det["emoji"]} {det["label"]} {det["score"]:.2f}'
                
                # Dibujar rectángulo y etiqueta
                cv2.rectangle(frame, (x, y), (x + w, y + h), color,10)
                
                # Fondo para el texto
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), color, -1)
                
                # Texto
                cv2.putText(frame, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Convertir imagen procesada de vuelta a base64
            _, buf = cv2.imencode(".jpg", frame)
            processed_image_bytes = buf.tobytes()
            processed_image_data = base64.b64encode(processed_image_bytes).decode("utf-8")
            
            # Guardar imagen procesada
            await save_received_image(processed_image_data, client_id)
            
            # Confirmar recepción al remitente
            await manager.send_to_client(client_id, {
                "type": "image_received",
                "message": f"Imagen procesada con {len(detections)} rostro(s) detectado(s)"
            })
            
            # Retransmitir imagen procesada si se solicita
            if broadcast:
                await manager.broadcast_to_all( {
                    "type": "image",
                    "imageData": processed_image_data,
                    "imageType": image_type,
                    "senderId": client_id,
                    "timestamp": int(time.time() * 1000),
                    "emotion_info": {
                        "faces_detected": len(detections),
                        "emotions": [det["label"] for det in detections]
                    }
                })
                
                await manager.send_to_client(client_id, {
                    "type": "server_message",
                    "message": f"Imagen con análisis retransmitida a {len(manager.active_connections) - 1} clientes"
                })
            
            print(f"Imagen procesada de {client_id}, {len(detections)} rostros detectados, retransmitida: {broadcast}")
        
        else:
            await manager.send_to_client(client_id, {
                "type": "error",
                "message": "No se pudo decodificar la imagen"
            })
        
    except Exception as e:
        await manager.send_to_client(client_id, {
            "type": "error",
            "message": f"Error procesando imagen: {str(e)}"
        })

async def handle_generate_test_image(client_id: str, message: dict):
    """Genera y envía una imagen de prueba con rostros simulados"""
    try:
        # Crear imagen de prueba más realista
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (50, 50, 50)  # Fondo gris oscuro
        
        # Simular "rostros" con círculos
        face_positions = [(160, 120), (480, 120), (320, 300)]
        face_colors = [(255, 200, 200), (200, 255, 200), (200, 200, 255)]
        
        for i, (cx, cy) in enumerate(face_positions):
            # Dibujar "cara" circular
            cv2.circle(test_image, (cx, cy), 60, face_colors[i], -1)
            cv2.circle(test_image, (cx, cy), 60, (255, 255, 255), 2)
            
            # Simular "ojos"
            cv2.circle(test_image, (cx-20, cy-20), 8, (0, 0, 0), -1)
            cv2.circle(test_image, (cx+20, cy-20), 8, (0, 0, 0), -1)
            
            # Simular "boca" - diferentes formas para diferentes emociones
            if i == 0:  # Feliz
                cv2.ellipse(test_image, (cx, cy+15), (25, 15), 0, 0, 180, (0, 0, 0), 2)
            elif i == 1:  # Triste
                cv2.ellipse(test_image, (cx, cy+25), (25, 15), 0, 180, 360, (0, 0, 0), 2)
            else:  # Neutral
                cv2.line(test_image, (cx-20, cy+20), (cx+20, cy+20), (0, 0, 0), 2)
        
        # Agregar texto informativo
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(test_image, f"Test PyTorch desde {client_id}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(test_image, timestamp, (50, 420), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(test_image, "Rostros simulados para test", (50, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Convertir a base64
        image_base64 = numpy_to_base64(test_image)
        
        # Determinar si es broadcast
        broadcast = message.get("broadcast", True)
        
        if broadcast:
            await manager.send_image_to_all(image_base64, "image/jpeg", client_id)
            await manager.send_to_client(client_id, {
                "type": "server_message",
                "message": "Imagen de prueba con rostros simulados enviada a todos"
            })
        else:
            await manager.send_image_to_client(client_id, image_base64, "image/jpeg", "server")
            await manager.send_to_client(client_id, {
                "type": "server_message",
                "message": "Imagen de prueba enviada solo a ti"
            })
        
    except Exception as e:
        await manager.send_to_client(client_id, {
            "type": "error",
            "message": f"Error generando imagen de prueba: {str(e)}"
        })

async def handle_server_test_image(client_id: str):
    """Envía una imagen de prueba del servidor con información del modelo"""
    try:
        # Crear imagen informativa del modelo
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (30, 30, 60)  # Fondo azul oscuro
        
        # Información del modelo
        model_info = [
            "PyTorch Emotion Recognition",
            "Hybrid Model: ShuffleNet + EfficientViT",
            "Cross-Attention Mechanism",
            f"Device: {emotion_recognizer.device}",
            f"Classes: {len(emotion_recognizer.class_names)}",
            "",
            "Emotions: " + ", ".join(emotion_recognizer.class_names),
            "",
            time.strftime("%Y-%m-%d %H:%M:%S")
        ]
        
        y_offset = 50
        for i, line in enumerate(model_info):
            if line:  # No dibujar líneas vacías
                color = (255, 255, 255) if i < 5 else (200, 200, 200)
                font_scale = 0.8 if i < 5 else 0.6
                cv2.putText(test_image, line, (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            y_offset += 35
        
        # Agregar un indicador visual del estado del modelo
        model_status = "LOADED" if emotion_recognizer.model is not None else "ERROR"
        status_color = (0, 255, 0) if emotion_recognizer.model is not None else (0, 0, 255)
        cv2.putText(test_image, f"Model Status: {model_status}", (50, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        
        # Convertir y enviar solo al solicitante
        image_base64 = numpy_to_base64(test_image)
        await manager.send_image_to_client(client_id, image_base64, "image/jpeg", "server")
        
    except Exception as e:
        await manager.send_to_client(client_id, {
            "type": "error",
            "message": f"Error enviando imagen del servidor: {str(e)}"
        })

# Variables globales para control de imágenes continuas
continuous_tasks = {}

async def handle_start_continuous(client_id: str):
    """Inicia el envío continuo de imágenes con datos de prueba"""
    if client_id in continuous_tasks:
        await manager.send_to_client(client_id, {
            "type": "server_message",
            "message": "Ya tienes imágenes continuas activas"
        })
        return
    
    async def send_continuous_images():
        counter = 0
        emotions_cycle = list(EMOTIONS.values())
        
        while client_id in continuous_tasks:
            try:
                # Crear imagen con datos dinámicos
                img = np.random.randint(50, 100, (240, 320, 3), dtype=np.uint8)
                
                # Simular datos de emoción
                current_emotion = emotions_cycle[counter % len(emotions_cycle)]
                
                # Agregar información
                cv2.putText(img, f"Frame #{counter}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, time.strftime("%H:%M:%S"), (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img, f"Emotion: {current_emotion['emoji']} {current_emotion['name']}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img, f"PyTorch Model Active", (10, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                image_base64 = numpy_to_base64(img)
                await manager.send_image_to_client(client_id, image_base64, "image/jpeg", "server_continuous")
                
                counter += 1
                await asyncio.sleep(3)  # Enviar cada 3 segundos
                
            except Exception as e:
                print(f"Error en imágenes continuas para {client_id}: {e}")
                break
        
        # Limpiar tarea
        if client_id in continuous_tasks:
            del continuous_tasks[client_id]
    
    # Iniciar tarea
    continuous_tasks[client_id] = asyncio.create_task(send_continuous_images())
    
    await manager.send_to_client(client_id, {
        "type": "server_message",
        "message": "Imágenes continuas iniciadas con modelo PyTorch (cada 3 segundos)"
    })

async def handle_stop_continuous(client_id: str):
    """Detiene el envío continuo de imágenes"""
    if client_id in continuous_tasks:
        continuous_tasks[client_id].cancel()
        del continuous_tasks[client_id]
        
        await manager.send_to_client(client_id, {
            "type": "server_message",
            "message": "Imágenes continuas detenidas"
        })
    else:
        await manager.send_to_client(client_id, {
            "type": "server_message",
            "message": "No tienes imágenes continuas activas"
        })

async def handle_status_request(client_id: str):
    """Envía el estado del servidor al cliente"""
    received_count = 0
    if os.path.exists("received_images"):
        files = os.listdir("received_images")
        received_count = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
    
    status = {
        "type": "server_status",
        "data": {
            "activeConnections": len(manager.active_connections),
            "clientId": client_id,
            "receivedImagesCount": received_count,
            "continuousActive": client_id in continuous_tasks,
            "serverTime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": {
                "type": "PyTorch Hybrid Model",
                "architectures": ["ShuffleNet V2", "EfficientViT-M2"],
                "device": str(emotion_recognizer.device),
                "loaded": emotion_recognizer.model is not None,
                "yolo_loaded": emotion_recognizer.yolo is not None,
                "classes": emotion_recognizer.class_names
            }
        }
    }
    
    await manager.send_to_client(client_id, status)

async def save_received_image(base64_data: str, sender_id: str):
    """Guarda una imagen recibida en el sistema de archivos"""
    try:
        os.makedirs("received_images", exist_ok=True)
        
        image_bytes = base64.b64decode(base64_data)
        timestamp = int(time.time())
        filename = f"received_images/pytorch_analysis_{sender_id}_{timestamp}.jpg"
        
        with open(filename, "wb") as f:
            f.write(image_bytes)
            
        print(f"Imagen con análisis PyTorch guardada: {filename}")
        
    except Exception as e:
        print(f"Error guardando imagen: {e}")

# Endpoints REST para monitoreo
@app.get("/status")
async def get_status():
    """Obtiene el estado del servidor con información del modelo PyTorch"""
    received_count = 0
    if os.path.exists("received_images"):
        files = os.listdir("received_images")
        received_count = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
    
    return {
        "active_connections": len(manager.active_connections),
        "server_running": True,
        "websocket_url": "ws://192.168.201.17:8080/ws",
        "received_images_count": received_count,
        "continuous_tasks_active": len(continuous_tasks),
        "model_info": {
            "type": "PyTorch Hybrid Emotion Recognition",
            "architectures": ["ShuffleNet V2", "EfficientViT-M2 with Cross-Attention"],
            "device": str(emotion_recognizer.device),
            "model_loaded": emotion_recognizer.model is not None,
            "yolo_loaded": emotion_recognizer.yolo is not None,
            "emotion_classes": emotion_recognizer.class_names,
            "model_path": emotion_recognizer.model_path
        },
        "features": {
            "websocket_only": True,
            "pytorch_inference": True,
            "emotion_recognition": True,
            "face_detection": True,
            "real_time_processing": True,
            "send_images": True,
            "receive_images": True,
            "broadcast": True,
            "continuous_images": True,
            "image_storage": True
        }
    }

@app.get("/received-images/{filename}")
async def get_received_image(filename: str):
    """Obtiene una imagen procesada específica"""
    file_path = f"received_images/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"error": "Imagen no encontrada"}

@app.get("/model-info")
async def get_model_info():
    """Obtiene información detallada del modelo PyTorch"""
    return {
        "model_architecture": "Hybrid CNN-Transformer",
        "backbones": ["ShuffleNet V2", "EfficientViT-M2"],
        "fusion_method": "Cross-Attention",
        "device": str(emotion_recognizer.device),
        "model_loaded": emotion_recognizer.model is not None,
        "yolo_model": emotion_recognizer.yolo_path,
        "emotion_classes": emotion_recognizer.class_names,
        "total_classes": len(emotion_recognizer.class_names),
        "input_size": "224x224",
        "preprocessing": "ImageNet normalization"
    }

@app.on_event("startup")
async def startup_event():
    """Inicializar servidor con modelo PyTorch"""
    print("=== SERVIDOR WEBSOCKET CON PYTORCH INICIADO ===")
    print("WebSocket URL: ws://192.168.201.17:8080/ws")
    print("Página de prueba: http://192.168.201.17:8080")
    print(f"Modelo PyTorch: {'✓ Cargado' if emotion_recognizer.model else '✗ Error'}")
    print(f"YOLO: {'✓ Cargado' if emotion_recognizer.yolo else '✗ Error'}")
    print(f"Dispositivo: {emotion_recognizer.device}")
    print("Análisis de emociones en tiempo real habilitado")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpia las tareas al cerrar el servidor"""
    # Cancelar todas las tareas continuas
    for client_id, task in continuous_tasks.items():
        task.cancel()
    continuous_tasks.clear()
    print("Servidor PyTorch detenido y tareas limpiadas")

if __name__ == "__main__":
    # Ejecutar servidor
    uvicorn.run(
        app, 
        host="192.168.201.17",  # IP específica
        port=8080,
        log_level="info"
    )