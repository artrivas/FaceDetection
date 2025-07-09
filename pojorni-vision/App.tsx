import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Dimensions,
  SafeAreaView,
  StatusBar,
  ScrollView,
  Image
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import * as FileSystem from 'expo-file-system';

const { width, height } = Dimensions.get('window');

export default function App() {
  const [currentImage, setCurrentImage] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [wsUrl, setWsUrl] = useState('ws://192.168.40.17:8080/ws');
  const [imageCount, setImageCount] = useState(0);
  const [sentImageCount, setSentImageCount] = useState(0);
  const [ws, setWs] = useState(null);
  const [clientId, setClientId] = useState(null);
  
  // Estados para streaming
  const [isStreaming, setIsStreaming] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraType, setCameraType] = useState('back');
  const [streamingFPS, setStreamingFPS] = useState(10);
  const [streamingQuality, setStreamingQuality] = useState(0.6);
  
  // Buffer para imágenes recibidas
  const [imageBuffer, setImageBuffer] = useState([]);
  
  const cameraRef = useRef(null);
  const streamingIntervalRef = useRef(null);

  // Solicitar permisos de cámara al cargar
  useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, []);

  // Efecto para streaming fluido sin animaciones
  useEffect(() => {
    if (imageBuffer.length > 0) {
      // Mostrar inmediatamente la última imagen sin animaciones
      const latestImage = imageBuffer[imageBuffer.length - 1];
      setCurrentImage(latestImage);
    } else {
      setCurrentImage(null);
    }
  }, [imageBuffer]);

  const connectWebSocket = () => {
    try {
      const websocket = new WebSocket(wsUrl);
      
      websocket.onopen = () => {
        setIsConnected(true);
        setWs(websocket);
        console.log('WebSocket conectado');
      };

      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'connection_established') {
            setClientId(data.clientId);
            console.log('Cliente conectado como:', data.clientId);
          } else if (data.type === 'image' && data.imageData) {
            // Solo mostrar imágenes que NO vengan de este cliente
            if (data.senderId !== clientId) {
              const newImage = {
                imageData: data.imageData,
                timestamp: Date.now(),
                type: data.imageType || 'image/jpeg',
                senderId: data.senderId
              };
              
              // Reemplazar inmediatamente sin buffer para evitar animaciones
              setCurrentImage(newImage);
              setImageCount(prev => prev + 1);
            }
          } else if (data.type === 'image_received') {
            // Remover console.log para mejor rendimiento
          } else if (data.type === 'server_message') {
            console.log('Mensaje del servidor:', data.message);
          } else if (data.type === 'error') {
            console.error('Error del servidor:', data.message);
          }
        } catch (error) {
          console.error('Error parsing message:', error);
        }
      };

      websocket.onclose = () => {
        setIsConnected(false);
        setWs(null);
        setClientId(null);
        stopStreaming();
        console.log('WebSocket desconectado');
      };

      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        Alert.alert('Error', 'Error de conexión WebSocket');
      };
    } catch (error) {
      Alert.alert('Error', 'No se pudo conectar al WebSocket');
    }
  };

  const disconnectWebSocket = () => {
    stopStreaming();
    if (ws) {
      ws.close();
    }
  };

  // Función optimizada para capturar y enviar frames
  const captureAndSendFrame = async () => {
    if (!cameraRef.current || !ws || !isConnected) {
      return;
    }

    try {
      // Configuración optimizada para máxima velocidad
      const photo = await cameraRef.current.takePictureAsync({
        quality: streamingQuality,
        base64: true,
        skipProcessing: true,
        imageType: 'jpg',
        exif: false,
        additionalExif: {},
        onPictureSaved: undefined,
        // Configuraciones adicionales para mejor rendimiento
        mirror: false,
        orientation: 'portrait',
        pauseAfterCapture: false,
        writeExif: false,
      });

      // Envío optimizado sin confirmación
      const message = {
        type: 'send_image',
        imageData: photo.base64,
        imageType: 'image/jpeg',
        broadcast: true,
        streaming: true,
        timestamp: Date.now()
      };

      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
        setSentImageCount(prev => prev + 1);
      }
      
    } catch (error) {
      // Silenciar errores para no interrumpir el flujo
      if (error.message.includes('concurrent')) {
        // Error común cuando se toman fotos muy rápido
        return;
      }
      console.warn('Frame skip:', error.message);
    }
  };

  // Iniciar streaming optimizado
  const startStreaming = () => {
    if (!permission?.granted) {
      Alert.alert('Error', 'No tienes permisos de cámara');
      return;
    }

    if (!isConnected) {
      Alert.alert('Error', 'Debes estar conectado para hacer streaming');
      return;
    }

    setIsStreaming(true);
    
    // Intervalo optimizado con mejor control de tiempo
    const intervalMs = Math.max(50, 1000 / streamingFPS); // Reducir mínimo a 50ms
    
    // Función recursiva optimizada para mejor timing
    const scheduleNextFrame = () => {
      const startTime = Date.now();
      
      if (streamingIntervalRef.current) {
        captureAndSendFrame()
          .finally(() => {
            if (streamingIntervalRef.current) {
              const elapsed = Date.now() - startTime;
              const nextDelay = Math.max(0, intervalMs - elapsed);
              streamingIntervalRef.current = setTimeout(scheduleNextFrame, nextDelay);
            }
          });
      }
    };

    streamingIntervalRef.current = setTimeout(scheduleNextFrame, 0);
    console.log(`Streaming HD HORIZONTAL iniciado a ${streamingFPS} FPS (1920x1080)`);
  };

  // Detener streaming
  const stopStreaming = () => {
    if (streamingIntervalRef.current) {
      clearTimeout(streamingIntervalRef.current);
      streamingIntervalRef.current = null;
    }
    setIsStreaming(false);
    console.log('Streaming detenido');
  };

  // Cambiar cámara
  const flipCamera = () => {
    setCameraType(
      cameraType === 'back' ? 'front' : 'back'
    );
  };

  const clearStats = () => {
    setSentImageCount(0);
    setImageCount(0);
    setCurrentImage(null);
  };

  useEffect(() => {
    return () => {
      stopStreaming();
      if (ws) {
        ws.close();
      }
    };
  }, [ws]);

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  if (!permission) {
    return (
      <View style={styles.container}>
        <Text>Cargando permisos de cámara...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text>No hay acceso a la cámara</Text>
        <TouchableOpacity onPress={requestPermission}>
          <Text>Solicitar permisos</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <ScrollView
        contentContainerStyle={{ flexGrow: 1 }}
        nestedScrollEnabled
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Camera Streaming</Text>
          <View style={styles.statusContainer}>
            <View style={[
              styles.statusDot, 
              { backgroundColor: isConnected ? '#4CAF50' : '#F44336' }
            ]} />
            <Text style={styles.statusText}>
              {isConnected ? `Conectado${clientId ? ` (${clientId})` : ''}` : 'Desconectado'}
            </Text>
          </View>
          
          {isStreaming && (
            <View style={styles.streamingIndicator}>
              <View style={styles.streamingDot} />
              <Text style={styles.streamingText}>
                📐 STREAMING HD HORIZONTAL ({streamingFPS} FPS)
              </Text>
            </View>
          )}
          
          <View style={styles.statsContainer}>
            {imageCount > 0 && (
              <Text style={styles.imageCount}>
                📥 Recibidas: {imageCount}
              </Text>
            )}
            {sentImageCount > 0 && (
              <Text style={styles.imageCount}>
                📤 Frames: {sentImageCount}
              </Text>
            )}
          </View>
        </View>

        {/* Controls */}
        <View style={styles.controls}>
          <TextInput
            style={styles.input}
            placeholder="WebSocket URL"
            value={wsUrl}
            onChangeText={setWsUrl}
            editable={!isConnected && !isStreaming}
          />
          
          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[
                styles.button,
                { backgroundColor: isConnected ? '#F44336' : '#4CAF50' }
              ]}
              onPress={isConnected ? disconnectWebSocket : connectWebSocket}
              disabled={isStreaming}
            >
              <Text style={styles.buttonText}>
                {isConnected ? 'Desconectar' : 'Conectar'}
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[styles.button, styles.clearButton]}
              onPress={clearStats}
              disabled={isStreaming}
            >
              <Text style={styles.buttonText}>Limpiar</Text>
            </TouchableOpacity>
          </View>

          {/* Streaming Controls */}
          {isConnected && (
            <>
              <View style={styles.streamingControls}>
                <TouchableOpacity
                  style={[
                    styles.streamingButton,
                    { backgroundColor: isStreaming ? '#F44336' : '#4CAF50' }
                  ]}
                  onPress={isStreaming ? stopStreaming : startStreaming}
                >
                  <Text style={styles.buttonText}>
                    {isStreaming ? '⏹ Detener Stream' : '📐 Stream HD Horizontal'}
                  </Text>
                </TouchableOpacity>
                
                <TouchableOpacity
                  style={[styles.smallButton, styles.flipButton]}
                  onPress={flipCamera}
                  disabled={isStreaming}
                >
                  <Text style={styles.buttonText}>🔄</Text>
                </TouchableOpacity>
              </View>

              {/* Settings */}
              <View style={styles.settingsContainer}>
                <View style={styles.settingRow}>
                  <Text style={styles.settingLabel}>FPS: {streamingFPS}</Text>
                  <View style={styles.fpsButtons}>
                    <TouchableOpacity
                      style={[styles.fpsButton, streamingFPS === 5 && styles.activeButton]}
                      onPress={() => setStreamingFPS(5)}
                      disabled={isStreaming}
                    >
                      <Text style={styles.fpsButtonText}>5</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.fpsButton, streamingFPS === 10 && styles.activeButton]}
                      onPress={() => setStreamingFPS(10)}
                      disabled={isStreaming}
                    >
                      <Text style={styles.fpsButtonText}>10</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.fpsButton, streamingFPS === 15 && styles.activeButton]}
                      onPress={() => setStreamingFPS(15)}
                      disabled={isStreaming}
                    >
                      <Text style={styles.fpsButtonText}>15</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.fpsButton, streamingFPS === 20 && styles.activeButton]}
                      onPress={() => setStreamingFPS(20)}
                      disabled={isStreaming}
                    >
                      <Text style={styles.fpsButtonText}>20</Text>
                    </TouchableOpacity>
                  </View>
                </View>
                
                <View style={styles.settingRow}>
                  <Text style={styles.settingLabel}>Calidad: {Math.round(streamingQuality * 100)}%</Text>
                  <View style={styles.fpsButtons}>
                    <TouchableOpacity
                      style={[styles.fpsButton, streamingQuality === 0.4 && styles.activeButton]}
                      onPress={() => setStreamingQuality(0.4)}
                      disabled={isStreaming}
                    >
                      <Text style={styles.fpsButtonText}>40%</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.fpsButton, streamingQuality === 0.6 && styles.activeButton]}
                      onPress={() => setStreamingQuality(0.6)}
                      disabled={isStreaming}
                    >
                      <Text style={styles.fpsButtonText}>60%</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[styles.fpsButton, streamingQuality === 0.8 && styles.activeButton]}
                      onPress={() => setStreamingQuality(0.8)}
                      disabled={isStreaming}
                    >
                      <Text style={styles.fpsButtonText}>80%</Text>
                    </TouchableOpacity>
                  </View>
                </View>
              </View>
            </>
          )}
        </View>

        {/* Hidden Camera optimizada */}
        <View style={styles.hiddenCameraContainer}>
          <CameraView
            ref={cameraRef}
            animateShutter={false}
            style={styles.hiddenCamera}
            facing={cameraType}
            mode="picture"
            videoQuality="1080p"
            pictureSize="1920x1080"
            ratio="16:9"
            animationsEnabled={false}
            enableTorch={false}
            responsiveOrientationWhenOrientationLocked={true}
            barcodeScannerSettings={{
              barcodeTypes: [],
            }}
            // Configuraciones adicionales para mejor rendimiento
            autoFocus="on"
            whiteBalance="auto"
            flashMode="off"
          />
          <View style={styles.cameraOverlay}>
            <Text style={styles.cameraText}>
              🔇 Cámara {cameraType === 'back' ? 'Trasera' : 'Frontal'} HORIZONTAL
            </Text>
            <Text style={styles.cameraSubtext}>
              {isStreaming ? `📐 HD ${streamingFPS} FPS (1920x1080)` : '📐 Lista para streaming HD horizontal'}
            </Text>
          </View>
        </View>

        {/* Display de imágenes recibidas - MÁS GRANDE Y SIN ANIMACIONES */}
        <View style={styles.imageContainer}>
          {!currentImage ? (
            <View style={styles.emptyContainer}>
              <Text style={styles.emptyIcon}>📺</Text>
              <Text style={styles.emptyText}>
                {isConnected 
                  ? 'Esperando imágenes de otros clientes...\n\n📐 Tu streaming es HD HORIZONTAL (1920x1080)' 
                  : 'Conéctate para streaming HD horizontal'
                }
              </Text>
            </View>
          ) : (
            <View style={styles.currentImageContainer}>
              <Image
                key={`img-${currentImage.timestamp}-${Math.random()}`}
                source={{ uri: `data:${currentImage.type};base64,${currentImage.imageData}` }}
                style={styles.currentImage}
                resizeMode="contain"
              />
              
              <View style={styles.imageInfo}>
                <Text style={styles.imageInfoText}>
                  📡 De: {currentImage.senderId}
                </Text>
                <Text style={styles.imageInfoText}>
                  🕒 Recibida: {formatTimestamp(currentImage.timestamp)}
                </Text>
              </View>
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: 'white',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  statusText: {
    fontSize: 14,
    fontWeight: '500',
  },
  streamingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  streamingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#4CAF50',
    marginRight: 8,
  },
  streamingText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#4CAF50',
  },
  statsContainer: {
    flexDirection: 'row',
    gap: 15,
  },
  imageCount: {
    fontSize: 12,
    color: '#666',
  },
  controls: {
    backgroundColor: 'white',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    marginBottom: 15,
    fontSize: 16,
  },
  buttonContainer: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 15,
  },
  button: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  clearButton: {
    backgroundColor: '#FF9800',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  streamingControls: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 15,
  },
  streamingButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  smallButton: {
    width: 50,
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  flipButton: {
    backgroundColor: '#9C27B0',
  },
  settingsContainer: {
    backgroundColor: '#f8f8f8',
    padding: 15,
    borderRadius: 8,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  settingLabel: {
    fontSize: 14,
    fontWeight: '500',
  },
  fpsButtons: {
    flexDirection: 'row',
    gap: 5,
  },
  fpsButton: {
    backgroundColor: '#ddd',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 4,
  },
  activeButton: {
    backgroundColor: '#2196F3',
  },
  fpsButtonText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#333',
  },
  hiddenCameraContainer: {
    height: 60, // Reducido aún más para dar más espacio a las imágenes
    backgroundColor: '#000',
    position: 'relative',
  },
  hiddenCamera: {
    flex: 1,
    opacity: 0.3,
  },
  cameraOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
  },
  cameraText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  cameraSubtext: {
    color: '#ccc',
    fontSize: 10,
    textAlign: 'center',
    marginTop: 2,
  },
  imageContainer: {
    flex: 1,
    minHeight: height * 0.7, // Aumentado para imagen más grande
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyIcon: {
    fontSize: 64,
    marginBottom: 20,
  },
  emptyText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    lineHeight: 24,
  },
  currentImage: {
    width: '100%',
    height: height * 0.95, // Imagen más grande
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#4CAF50',
  },
  imageInfo: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 8,
    marginTop: 10,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  imageInfoText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 5,
  },
});