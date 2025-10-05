"""
ASL Recognition Pipeline: WebSocket → MediaPipe → ASL.h5 → (1, 29) output vector
Complete integration for real-time ASL recognition from webcam frames.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import asyncio
import websockets
import json
import base64
import os
from io import BytesIO
from PIL import Image
import logging
from typing import Optional, Dict, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASLPipeline:
    """Complete ASL recognition pipeline from webcam frames to predictions"""
    
    def __init__(self, model_path: str = "ASL.h5"):
        """Initialize the ASL pipeline with MediaPipe and ASL model"""
        
        # ASL class labels (29 classes: A-Z, space, del, nothing)
        self.class_labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'space', 'del', 'nothing'
        ]
        
        # Load ASL model
        logger.info(f"Loading ASL model from {model_path}...")
        self.asl_model = keras.models.load_model(model_path)
        logger.info(f"ASL model loaded: {self.asl_model.input_shape} -> {self.asl_model.output_shape}")
        
        # Initialize MediaPipe
        logger.info("Initializing MediaPipe...")
        self._init_mediapipe()
        
        logger.info("ASL Pipeline initialized successfully!")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe hands detection"""
        try:
            # Try new MediaPipe API first
            import mediapipe.tasks.python as mp_tasks
            import mediapipe.tasks.python.vision as mp_vision
            
            # Download hand landmark model if not exists
            model_path = "models/hand_landmarker.task"
            if not os.path.exists(model_path):
                self._download_mediapipe_model()
            
            # Create hand landmarker
            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.5
            )
            self.hand_detector = mp_vision.HandLandmarker.create_from_options(options)
            self.api_type = "new"
            logger.info("Using new MediaPipe API")
            
        except ImportError:
            # Fall back to legacy MediaPipe API
            self.mp_hands = mp.solutions.hands
            self.hand_detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.api_type = "legacy"
            logger.info("Using legacy MediaPipe API")
    
    def _download_mediapipe_model(self):
        """Download MediaPipe hand landmark model"""
        import os
        import urllib.request
        
        os.makedirs("models", exist_ok=True)
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        model_path = "models/hand_landmarker.task"
        
        if not os.path.exists(model_path):
            logger.info("Downloading MediaPipe hand landmark model...")
            urllib.request.urlretrieve(model_url, model_path)
            logger.info(f"Model downloaded to {model_path}")
    
    def detect_hand_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect hand region in image and return bounding box coordinates
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (x, y, width, height) or None if no hand detected
        """
        try:
            if self.api_type == "new":
                # Convert BGR to RGB for MediaPipe
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # Detect hands
                detection_result = self.hand_detector.detect(mp_image)
                
                if detection_result.hand_landmarks:
                    # Get first hand landmarks
                    hand_landmarks = detection_result.hand_landmarks[0]
                    
                    # Calculate bounding box from landmarks
                    h, w = image.shape[:2]
                    x_coords = [lm.x * w for lm in hand_landmarks]
                    y_coords = [lm.y * h for lm in hand_landmarks]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    return (x_min, y_min, x_max - x_min, y_max - y_min)
            
            else:
                # Legacy API
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hand_detector.process(rgb_image)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Calculate bounding box
                    h, w = image.shape[:2]
                    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)
                    
                    return (x_min, y_min, x_max - x_min, y_max - y_min)
            
            return None
            
        except Exception as e:
            logger.error(f"Hand detection error: {e}")
            return None
    
    def preprocess_for_asl_model(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Preprocess detected hand region for ASL model input
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            
        Returns:
            Preprocessed image ready for ASL model (1, 64, 64, 3)
        """
        x, y, w, h = bbox
        
        # Extract hand region
        hand_roi = image[y:y+h, x:x+w]
        
        # Resize to 64x64 (ASL model input size)
        hand_resized = cv2.resize(hand_roi, (64, 64))
        
        # Normalize pixel values to [0, 1]
        hand_normalized = hand_resized.astype(np.float32) / 255.0
        
        # Add batch dimension: (64, 64, 3) -> (1, 64, 64, 3)
        hand_batch = np.expand_dims(hand_normalized, axis=0)
        
        return hand_batch
    
    def predict_asl_sign(self, processed_image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Predict ASL sign from preprocessed image
        
        Args:
            processed_image: Preprocessed image (1, 64, 64, 3)
            
        Returns:
            Tuple of (prediction_vector, predicted_class, confidence)
        """
        # Get model prediction - this is our (1, 29) vector!
        prediction_vector = self.asl_model.predict(processed_image, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(prediction_vector[0])
        confidence = float(prediction_vector[0][predicted_class_idx])
        predicted_class = self.class_labels[predicted_class_idx]
        
        return prediction_vector, predicted_class, confidence
    
    def process_frame(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Complete pipeline: process a single frame and return prediction
        
        Args:
            image: Input image from webcam
            
        Returns:
            Dictionary with prediction results
        """
        result = {
            "hand_detected": False,
            "prediction_vector": None,
            "predicted_sign": None,
            "confidence": 0.0,
            "api_used": self.api_type,
            "error": None
        }
        
        try:
            # Step 1: Detect hand region with MediaPipe
            bbox = self.detect_hand_region(image)
            
            if bbox is None:
                result["error"] = "No hand detected"
                return result
            
            result["hand_detected"] = True
            
            # Step 2: Preprocess for ASL model
            processed_image = self.preprocess_for_asl_model(image, bbox)
            
            # Step 3: Get ASL prediction - THE (1, 29) VECTOR!
            prediction_vector, predicted_class, confidence = self.predict_asl_sign(processed_image)
            
            # Step 4: Return results
            result["prediction_vector"] = prediction_vector.tolist()  # Convert to list for JSON
            result["predicted_sign"] = predicted_class
            result["confidence"] = confidence
            
            logger.info(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Pipeline error: {e}")
        
        return result


class ASLWebSocketServer:
    """WebSocket server for real-time ASL recognition"""
    
    def __init__(self, host: str = "localhost", port: int = 4000):
        self.host = host
        self.port = port
        self.pipeline = ASLPipeline()
    
    async def handle_client(self, websocket, path):
        """Handle incoming WebSocket connections"""
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "webcam_frame":
                        # Process webcam frame
                        response = await self.process_webcam_frame(data)
                        await websocket.send(json.dumps(response))
                    
                    elif data.get("type") == "test":
                        # Test message
                        response = {
                            "type": "test_response",
                            "message": "ASL Pipeline server is running!",
                            "timestamp": data.get("timestamp")
                        }
                        await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError:
                    error_response = {
                        "type": "error",
                        "message": "Invalid JSON format"
                    }
                    await websocket.send(json.dumps(error_response))
                
                except Exception as e:
                    error_response = {
                        "type": "error",
                        "message": f"Processing error: {str(e)}"
                    }
                    await websocket.send(json.dumps(error_response))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def process_webcam_frame(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming webcam frame through ASL pipeline"""
        
        try:
            # Decode base64 frame data
            frame_data = data["frame_data"]
            
            # Remove data URL prefix if present
            if frame_data.startswith("data:image"):
                frame_data = frame_data.split(",")[1]
            
            # Decode base64 to image
            image_bytes = base64.b64decode(frame_data)
            image_pil = Image.open(BytesIO(image_bytes))
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
            # Process through ASL pipeline
            result = self.pipeline.process_frame(image_cv)
            
            # Format response
            response = {
                "type": "asl_prediction",
                "timestamp": data.get("timestamp"),
                **result
            }
            
            return response
            
        except Exception as e:
            return {
                "type": "asl_prediction",
                "error": f"Frame processing failed: {str(e)}",
                "hand_detected": False
            }
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting ASL WebSocket server on {self.host}:{self.port}")
        logger.info("Pipeline ready for real-time ASL recognition!")
        logger.info("Expected input: WebSocket frames → MediaPipe → ASL.h5 → (1, 29) vector output")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info("Server started! Waiting for connections...")
            await asyncio.Future()  # Run forever


def main():
    # Create and start server
    server = ASLWebSocketServer()
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server error: {e}")


if __name__ == "__main__":
    main()