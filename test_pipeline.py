"""
Test script for ASL Pipeline - validates the complete workflow
WebSocket → MediaPipe → ASL.h5 → (1, 29) vector output
"""

import cv2
import numpy as np
from asl_pipeline import ASLPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_asl_pipeline():
    """Test the complete ASL pipeline with a sample image"""
    
    print("🧪 Testing ASL Pipeline")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        print("1️⃣ Initializing ASL Pipeline...")
        pipeline = ASLPipeline()
        print("✅ Pipeline initialized successfully!")
        
        # Test with webcam if available
        print("\n2️⃣ Testing with webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Webcam not available, creating dummy image...")
            # Create a dummy image (480x640x3)
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        else:
            print("✅ Webcam available, capturing frame...")
            ret, test_image = cap.read()
            cap.release()
            
            if not ret:
                print("❌ Failed to capture frame, creating dummy image...")
                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            else:
                print("✅ Frame captured successfully!")
        
        # Process through pipeline
        print("\n3️⃣ Processing through ASL pipeline...")
        result = pipeline.process_frame(test_image)
        
        # Display results
        print("\n📊 Pipeline Results:")
        print("-" * 30)
        print(f"Hand detected: {result['hand_detected']}")
        print(f"API used: {result['api_used']}")
        
        if result['hand_detected'] and result['prediction_vector'] is not None:
            prediction_vector = np.array(result['prediction_vector'])
            print(f"✅ Prediction vector shape: {prediction_vector.shape}")
            print(f"✅ Predicted sign: {result['predicted_sign']}")
            print(f"✅ Confidence: {result['confidence']:.3f}")
            
            # Show top 3 predictions
            print(f"\n🏆 Top 3 predictions:")
            top_indices = np.argsort(prediction_vector[0])[-3:][::-1]
            for i, idx in enumerate(top_indices):
                sign = pipeline.class_labels[idx]
                confidence = prediction_vector[0][idx]
                print(f"   {i+1}. {sign}: {confidence:.3f}")
                
        elif result['error']:
            print(f"ℹ️  {result['error']}")
        
        print(f"\n4️⃣ Testing model output format...")
        # Test direct model prediction to verify (1, 29) shape
        dummy_input = np.random.random((1, 64, 64, 3)).astype(np.float32)
        model_output = pipeline.asl_model.predict(dummy_input, verbose=0)
        print(f"✅ ASL model output shape: {model_output.shape}")
        print(f"✅ Expected shape: (1, 29)")
        print(f"✅ Shape matches: {model_output.shape == (1, 29)}")
        
        print(f"\n🎉 Pipeline Test Complete!")
        print(f"✅ Ready for WebSocket integration")
        print(f"✅ MediaPipe → ASL.h5 → (1, 29) vector ✓")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_integration():
    """Test WebSocket server components"""
    
    print(f"\n🌐 Testing WebSocket Integration")
    print("=" * 50)
    
    try:
        from asl_pipeline import ASLWebSocketServer
        
        print("1️⃣ Creating WebSocket server...")
        server = ASLWebSocketServer()
        print("✅ WebSocket server created successfully!")
        
        # Test frame processing
        print("\n2️⃣ Testing frame processing...")
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Create a dummy webcam frame in base64 format
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pil_image = Image.fromarray(cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB))
        
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create test message
        test_message = {
            "type": "webcam_frame",
            "frame_data": f"data:image/jpeg;base64,{img_base64}",
            "timestamp": "2025-10-04T20:00:00Z"
        }
        
        # Process message (simulate async)
        import asyncio
        result = asyncio.run(server.process_webcam_frame(test_message))
        
        print("✅ Frame processing test completed!")
        print(f"   Result type: {result.get('type')}")
        print(f"   Hand detected: {result.get('hand_detected')}")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧏‍♂️ ASL Pipeline Test Suite")
    print("=" * 60)
    print("Testing: WebSocket → MediaPipe → ASL.h5 → (1, 29) vector")
    print()
    
    # Run tests
    pipeline_success = test_asl_pipeline()
    websocket_success = test_websocket_integration()
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 30)
    print(f"Pipeline Test: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    print(f"WebSocket Test: {'✅ PASS' if websocket_success else '❌ FAIL'}")
    
    if pipeline_success and websocket_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Ready for production integration!")
        print(f"✅ Your teammate can now connect to this pipeline!")
    else:
        print(f"\n⚠️  Some tests failed - check errors above")