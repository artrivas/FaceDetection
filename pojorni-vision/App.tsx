import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  Image,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Dimensions,
  SafeAreaView,
  StatusBar,
  ScrollView
} from 'react-native';

const { width, height } = Dimensions.get('window');

export default function App() {
  const [currentImage, setCurrentImage] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [wsUrl, setWsUrl] = useState('ws://192.168.52.17:8080/ws');
  const [imageCount, setImageCount] = useState(0);
  const [ws, setWs] = useState(null);

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
          
          if (data.type === 'image' && data.imageData) {
            const newImage = {
              imageData: data.imageData,
              timestamp: Date.now(),
              type: data.imageType || 'image/jpeg'
            };
            
            setCurrentImage(newImage);
            setImageCount(prev => prev + 1);
          }
        } catch (error) {
          console.error('Error parsing message:', error);
        }
      };

      websocket.onclose = () => {
        setIsConnected(false);
        setWs(null);
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
    if (ws) {
      ws.close();
    }
  };

  const clearImage = () => {
    setCurrentImage(null);
    setImageCount(0);
  };

  useEffect(() => {
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [ws]);

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Live Image Viewer</Text>
        <View style={styles.statusContainer}>
          <View style={[
            styles.statusDot, 
            { backgroundColor: isConnected ? '#4CAF50' : '#F44336' }
          ]} />
          <Text style={styles.statusText}>
            {isConnected ? 'Conectado' : 'Desconectado'}
          </Text>
        </View>
        {imageCount > 0 && (
          <Text style={styles.imageCount}>
            Imágenes recibidas: {imageCount}
          </Text>
        )}
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <TextInput
          style={styles.input}
          placeholder="WebSocket URL"
          value={wsUrl}
          onChangeText={setWsUrl}
          editable={!isConnected}
        />
        
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[
              styles.button,
              { backgroundColor: isConnected ? '#F44336' : '#4CAF50' }
            ]}
            onPress={isConnected ? disconnectWebSocket : connectWebSocket}
          >
            <Text style={styles.buttonText}>
              {isConnected ? 'Desconectar' : 'Conectar'}
            </Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={[styles.button, styles.clearButton]}
            onPress={clearImage}
          >
            <Text style={styles.buttonText}>Limpiar</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Current Image Display */}
      <ScrollView style={styles.imageContainer} contentContainerStyle={styles.imageContent}>
        {!currentImage ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyIcon}>📷</Text>
            <Text style={styles.emptyText}>
              {isConnected 
                ? 'Esperando la primera imagen...' 
                : 'Conéctate para recibir imágenes'
              }
            </Text>
          </View>
        ) : (
          <View style={styles.currentImageContainer}>
            <Image
              source={{ uri: `data:${currentImage.type};base64,${currentImage.imageData}` }}
              style={styles.currentImage}
              resizeMode="contain"
            />
            
            {/* Image Info */}
            <View style={styles.imageInfo}>
              <Text style={styles.imageInfoText}>
                Tipo: {currentImage.type}
              </Text>
              <Text style={styles.imageInfoText}>
                Recibida: {formatTimestamp(currentImage.timestamp)}
              </Text>
            </View>
          </View>
        )}
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
    marginBottom: 5,
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
  imageContainer: {
    flex: 1,
  },
  imageContent: {
    flexGrow: 1,
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
  currentImageContainer: {
    flex: 1,
    padding: 20,
  },
  currentImage: {
    width: '100%',
    height: height * 0.5,
    backgroundColor: '#f0f0f0',
    borderRadius: 8,
  },
  imageInfo: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 8,
    marginTop: 15,
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