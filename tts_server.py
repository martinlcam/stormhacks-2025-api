import asyncio
import websockets
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your API key
ELEVENLABS_API_KEY = "sk_e9c8679697a1343d65ea2f0d018f12d5beb95a0b07674cf6"
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


async def stream_tts_to_client(text_generator_func, client_websocket):
    """
    Stream TTS audio from ElevenLabs to the client WebSocket.
    
    Args:
        text_generator_func: Function that generates text chunks
        client_websocket: WebSocket connection to the frontend client
    """
    # WebSocket URL for ElevenLabs with MAXIMUM latency optimization
    url = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id=eleven_turbo_v2_5&output_format=mp3_44100_128&optimize_streaming_latency=4"
    
    try:
        async with websockets.connect(url) as elevenlabs_ws:
            print("Connected to ElevenLabs WebSocket")
            
            # Send initial configuration for LIVE streaming
            config = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "speed": 0.92
                },
                "generation_config": {
                    "chunk_length_schedule": [50, 90, 120, 150]  # Lower thresholds = faster generation
                },
                "xi_api_key": ELEVENLABS_API_KEY
            }
            await elevenlabs_ws.send(json.dumps(config))
            
            # Create tasks for sending and receiving
            chunk_count = 0
            text_buffer = []  # Track all sent text chunks
            
            async def send_text():
                """Call text generator and send to ElevenLabs"""
                nonlocal chunk_count
                try:
                    loop = asyncio.get_event_loop()
                    while True:
                        # Call your function to get text
                        text = await loop.run_in_executor(None, text_generator_func)
                        
                        # If function returns None or empty, stop
                        if text is None or text == "":
                            print("Generator finished, flushing stream")
                            # Flush remaining buffered text
                            await elevenlabs_ws.send(json.dumps({"text": "", "flush": True}))
                            # Send end signal to client
                            await client_websocket.send_json({"type": "end"})
                            break
                        
                        chunk_count += 1
                        text_buffer.append(text)
                        
                        # Send text to ElevenLabs with LIVE streaming
                        # Trigger generation EVERY chunk for minimum latency
                        message = {
                            "text": text,
                            "try_trigger_generation": True  # Always trigger for live streaming
                        }
                        await elevenlabs_ws.send(json.dumps(message))
                        print(f"Sent to ElevenLabs (chunk {chunk_count}, LIVE): {text.strip()}")
                        
                except Exception as e:
                    print(f"Send error: {e}")
            
            async def receive_and_forward_audio():
                """Receive audio from ElevenLabs and forward to client"""
                try:
                    async for message in elevenlabs_ws:
                        data = json.loads(message)
                        
                        # Check if audio data is present
                        if "audio" in data:
                            # Forward base64 audio to client
                            await client_websocket.send_json({
                                "type": "audio",
                                "data": data["audio"]
                            })
                            print(f"Audio chunk sent, isFinal: {data.get('isFinal', False)}")
                        
                        # Check if this is the final chunk
                        if data.get("isFinal", False):
                            print("Audio stream complete")
                            
                except Exception as e:
                    print(f"Receive error: {e}")
            
            # Run send and receive concurrently
            await asyncio.gather(send_text(), receive_and_forward_audio())
            
    except Exception as e:
        print(f"WebSocket error: {e}")
        await client_websocket.send_json({"type": "error", "data": str(e)})


# Test text generator function - configurable chunk size
# ADJUST THIS to find the sweet spot (try 3, 4, 5, 6, 7, 8 words)
WORDS_PER_CHUNK = 4  # <-- Smaller chunks for LIVE streaming (lower latency)

test_words = [
    "Hello", "world,", "this", "is", "a", "test", "of", "the",
    "ElevenLabs", "text", "to", "speech", "streaming", "API.",
    "Each", "word", "is", "sent", "individually", "for",
    "real-time", "synchronization", "with", "audio.", "You", "can",
    "adjust", "the", "chunk", "size", "to", "find", "the",
    "perfect", "balance", "between", "latency", "and", "smoothness.",
    "Larger", "chunks", "mean", "smoother", "audio", "but", "slightly",
    "more", "delay.", "Try", "different", "values","!"
]
word_counter = 0

def test_text_generator():
    """
    Test function that generates text in configurable chunks.
    Adjust WORDS_PER_CHUNK above to experiment.
    """
    global word_counter
    
    if word_counter >= len(test_words):
        return None
    
    import time
    time.sleep(0.05)  # Ultra-fast generation for LIVE streaming
    
    # Get N words at a time (configurable)
    chunk = ""
    words_to_get = min(WORDS_PER_CHUNK, len(test_words) - word_counter)
    
    for i in range(words_to_get):
        chunk += test_words[word_counter] + " "
        word_counter += 1
    
    print(f"Generated chunk ({words_to_get} words): {chunk.strip()}")
    return chunk


@app.websocket("/ws/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for TTS streaming.
    Frontend connects here to receive audio.
    """
    await websocket.accept()
    print("Client connected")
    
    try:
        # Reset counter for new connection
        global word_counter
        word_counter = 0
        
        # Start streaming TTS
        await stream_tts_to_client(test_text_generator, websocket)
        
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        print("Client disconnected")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "TTS WebSocket Server", "websocket_endpoint": "/ws/tts"}


if __name__ == "__main__":
    print("Starting TTS WebSocket server...")
    print("WebSocket endpoint: ws://localhost:8000/ws/tts")
    print("Connect your frontend to receive audio streams")
    uvicorn.run(app, host="0.0.0.0", port=8000)
