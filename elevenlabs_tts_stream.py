import asyncio
import websockets
import json
import base64
import pyaudio
import sys

# Audio playback setup
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050

async def stream_tts_from_generator(text_generator_func, api_key, voice_id="21m00Tcm4TlvDq8ikWAM"):
    """
    Stream text to speech using ElevenLabs WebSocket API.
    Continuously calls your text_generator_func to get text chunks.
    
    Args:
        text_generator_func: A callable function that returns text chunks (or None to stop)
        api_key (str): Your ElevenLabs API key
        voice_id (str): Voice ID to use
    """
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    audio_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    
    # WebSocket URL
    url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_monolingual_v1"
    
    try:
        async with websockets.connect(url) as websocket:
            print("Connected to ElevenLabs WebSocket", file=sys.stderr)
            print("Calling your text generator function...", file=sys.stderr)
            
            # Send initial configuration
            config = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8
                },
                "xi_api_key": api_key
            }
            await websocket.send(json.dumps(config))
            
            # Create tasks for sending and receiving
            async def send_text():
                """Call text generator function and send to WebSocket"""
                try:
                    loop = asyncio.get_event_loop()
                    while True:
                        # Call your function to get text
                        text = await loop.run_in_executor(None, text_generator_func)
                        
                        # If function returns None or empty string, stop
                        if text is None or text == "":
                            print("Generator returned None/empty, closing stream", file=sys.stderr)
                            await websocket.send(json.dumps({"text": ""}))
                            break
                        
                        # Send text to API
                        message = {"text": text}
                        await websocket.send(json.dumps(message))
                        print(f"Sent: {text}", file=sys.stderr)
                        
                except Exception as e:
                    print(f"Send error: {e}", file=sys.stderr)
            
            async def receive_audio():
                """Receive and play audio from the WebSocket"""
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        
                        # Check if audio data is present
                        if "audio" in data:
                            # Decode base64 audio
                            audio_data = base64.b64decode(data["audio"])
                            # Play audio
                            audio_stream.write(audio_data)
                        
                        # Check if this is the final chunk
                        if data.get("isFinal", False):
                            print("Audio chunk complete", file=sys.stderr)
                            
                except Exception as e:
                    print(f"Receive error: {e}", file=sys.stderr)
            
            # Run send and receive concurrently
            await asyncio.gather(send_text(), receive_audio())
            
    except Exception as e:
        print(f"WebSocket error: {e}", file=sys.stderr)
    finally:
        # Cleanup
        audio_stream.stop_stream()
        audio_stream.close()
        p.terminate()
        print("\nConnection closed", file=sys.stderr)


# Example usage
if __name__ == "__main__":
    # API key
    api_key = "sk_f5032056e13c161bd72ed8718d34d1b6ce2d0a5baf26d04e"
    
    # Example: Your text generator function
    # This is where you put YOUR function that generates text
    counter = 0
    def my_text_generator():
        """
        This is YOUR function that generates text.
        Replace this with your actual text generation logic.
        Return None or "" to stop the stream.
        """
        global counter
        counter += 1
        
        if counter > 5:  # Stop after 5 calls
            return None
        
        # Your logic here - could be reading from a queue, calling an API, etc.
        import time
        time.sleep(1)  # Simulate some processing time
        return f"This is text chunk number {counter}. "
    
    print("Starting TTS stream from your generator function...", file=sys.stderr)
    asyncio.run(stream_tts_from_generator(my_text_generator, api_key))
