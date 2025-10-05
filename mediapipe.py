"""
MediaPipe Hand Landmarker → ASL Model Pipeline (Compatible Version)
Works with both old and new MediaPipe APIs for maximum compatibility
"""

import cv2
import numpy as np
from tensorflow import keras
from typing import Optional, Tuple, List
import urllib.request
import os

class ASLHandProcessor:
    def __init__(self, model_path: str = "models/ASL.h5", use_new_api: bool = True):
        """Initialize MediaPipe and ASL model"""
        
        self.use_new_api = use_new_api
        
        if use_new_api:
            self.setup_new_mediapipe_api()
        else:
            self.setup_legacy_mediapipe_api()
        
        # Load ASL model
        try:
            self.asl_model = keras.models.load_model(model_path)
            print(f"✅ ASL model loaded: {self.asl_model.input_shape}")
        except Exception as e:
            print(f"❌ Failed to load ASL model: {e}")
            self.asl_model = None
        
        # Class mapping
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                       'del', 'nothing', 'space']
    
    def setup_new_mediapipe_api(self):
        """Setup using the new MediaPipe Tasks API"""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Setup model (check local first, then download)
            model_path = self.setup_mediapipe_model()
            if model_path is None:
                raise Exception("Could not find or download hand landmarker model")
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            print(f"✅ Using new MediaPipe Tasks API with model: {model_path}")
            
        except ImportError as e:
            print(f"⚠️ New MediaPipe API not available: {e}")
            print("Falling back to legacy API...")
            self.use_new_api = False
            self.setup_legacy_mediapipe_api()
    
    def setup_legacy_mediapipe_api(self):
        """Setup using the legacy MediaPipe API"""
        try:
            import mediapipe as mp
            
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            print("✅ Using legacy MediaPipe API")
            
        except Exception as e:
            print(f"❌ Failed to initialize MediaPipe: {e}")
            self.hands = None
    
    def setup_mediapipe_model(self):
        """Setup MediaPipe hand landmarker model (check local first, then download)"""
        # Check multiple possible locations for the model file
        possible_paths = [
            'hand_landmarker.task',                    # Current directory
            'models/hand_landmarker.task',             # In models directory
            'mediapipe_models/hand_landmarker.task',   # In dedicated mediapipe folder
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ Found local MediaPipe model: {path}")
                break
        
        if model_path is None:
            # Download to models directory if it doesn't exist locally
            model_path = 'models/hand_landmarker.task'
            os.makedirs('models', exist_ok=True)
            
            if not os.path.exists(model_path):
                print("📥 Downloading MediaPipe hand landmarker model...")
                model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
                try:
                    urllib.request.urlretrieve(model_url, model_path)
                    print(f"✅ Downloaded {model_path}")
                except Exception as e:
                    print(f"❌ Failed to download model: {e}")
                    print("Please download manually from:")
                    print(model_url)
                    return None
        
        return model_path
    
    def calculate_bounding_box(self, landmarks, image_shape: Tuple[int, int], 
                              padding: float = 0.2) -> Optional[Tuple[int, int, int, int]]:
        """Calculate bounding box from hand landmarks (works with both APIs)"""
        if not landmarks:
            return None
            
        height, width = image_shape[:2]
        
        # Extract coordinates based on API type
        if self.use_new_api:
            # New API: landmarks is a list of landmark objects
            x_coords = [lm.x * width for lm in landmarks]
            y_coords = [lm.y * height for lm in landmarks]
        else:
            # Legacy API: landmarks.landmark is the list
            x_coords = [lm.x * width for lm in landmarks.landmark]
            y_coords = [lm.y * height for lm in landmarks.landmark]
        
        # Find bounding box
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        pad_x = int(bbox_width * padding)
        pad_y = int(bbox_height * padding)
        
        # Apply padding with bounds checking
        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(width, x_max + pad_x)
        y2 = min(height, y_max + pad_y)
        
        return (x1, y1, x2, y2)
    
    def preprocess_for_asl_model(self, hand_crop: np.ndarray) -> np.ndarray:
        """Preprocess cropped hand image for ASL model"""
        # Resize to model input size (64x64)
        resized = cv2.resize(hand_crop, (64, 64))
        
        # Convert BGR to RGB (if from OpenCV)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] range
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def predict_asl_sign(self, model_input: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Make ASL prediction using the model"""
        if self.asl_model is None:
            return "no_model", 0.0, np.zeros(29)
        
        # Model prediction
        predictions = self.asl_model.predict(model_input, verbose=0)
        
        # Get predicted class
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_class = self.classes[predicted_idx]
        
        return predicted_class, confidence, predictions[0]
    
    def process_frame(self, frame: np.ndarray) -> Optional[dict]:
        """Complete pipeline: webcam frame → ASL prediction"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.use_new_api:
            return self.process_frame_new_api(frame, rgb_frame)
        else:
            return self.process_frame_legacy_api(frame, rgb_frame)
    
    def process_frame_new_api(self, frame: np.ndarray, rgb_frame: np.ndarray) -> Optional[dict]:
        """Process frame using new MediaPipe API"""
        try:
            import mediapipe as mp
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # MediaPipe hand detection
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.hand_landmarks:
                return None
            
            # Get first hand landmarks
            hand_landmarks = detection_result.hand_landmarks[0]
            
        except Exception as e:
            print(f"⚠️ New API failed: {e}, falling back to legacy")
            self.use_new_api = False
            return self.process_frame_legacy_api(frame, rgb_frame)
        
        return self.process_landmarks(frame, hand_landmarks)
    
    def process_frame_legacy_api(self, frame: np.ndarray, rgb_frame: np.ndarray) -> Optional[dict]:
        """Process frame using legacy MediaPipe API"""
        if self.hands is None:
            return None
        
        # MediaPipe hand detection
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Get first hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        return self.process_landmarks(frame, hand_landmarks)
    
    def process_landmarks(self, frame: np.ndarray, hand_landmarks) -> Optional[dict]:
        """Process detected landmarks to get ASL prediction"""
        # Calculate bounding box
        bbox = self.calculate_bounding_box(hand_landmarks, frame.shape)
        
        if bbox is None:
            return None
        
        # Crop hand region
        x1, y1, x2, y2 = bbox
        hand_crop = frame[y1:y2, x1:x2]
        
        if hand_crop.size == 0:
            return None
        
        # Preprocess for ASL model
        model_input = self.preprocess_for_asl_model(hand_crop)
        
        # ASL prediction
        predicted_class, confidence, all_probs = self.predict_asl_sign(model_input)
        
        # Extract landmark coordinates for output
        if self.use_new_api:
            landmarks_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
        else:
            landmarks_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        
        return {
            'predicted_sign': predicted_class,
            'confidence': float(confidence),
            'bbox': bbox,
            'hand_crop_shape': hand_crop.shape,
            'all_probabilities': all_probs.tolist(),
            'landmarks': landmarks_list,
            'api_used': 'new' if self.use_new_api else 'legacy'
        }
    
    def cleanup(self):
        """Release MediaPipe resources"""
        if self.use_new_api and hasattr(self, 'detector'):
            self.detector.close()
        elif not self.use_new_api and hasattr(self, 'hands') and self.hands:
            self.hands.close()

# Usage example
def test_realtime_processing():
    """Test the ASL processor with webcam"""
    processor = ASLHandProcessor(use_new_api=True)  # Try new API first
    cap = cv2.VideoCapture(0)
    
    print("🎥 Starting real-time ASL recognition")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = processor.process_frame(frame)
            
            # Display results
            if result:
                # Draw bounding box
                x1, y1, x2, y2 = result['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Show prediction
                api_info = f"[{result['api_used'].upper()}]"
                text = f"{result['predicted_sign']} ({result['confidence']:.2f}) {api_info}"
                cv2.putText(frame, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                print(f"Detected: {result['predicted_sign']} (conf: {result['confidence']:.3f}) via {result['api_used']} API")
            
            # Show frame
            cv2.imshow('ASL Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        processor.cleanup()

if __name__ == "__main__":
    test_realtime_processing()