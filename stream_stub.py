"""Simple WebSocket stub to accept MediaRecorder binary chunks and reply with synthetic asl_batch messages.

Usage: python stream_stub.py

This server listens on ws://localhost:4000/ws/stream/ and logs incoming binary message sizes.
It responds with a JSON `asl_batch` message so the frontend can validate connectivity without installing heavy ML deps.
"""
import asyncio
import json
import logging
import time
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stream_stub")

HOST = 'localhost'
PORT = 4000
PATH = '/ws/stream/'

async def handle_stream(websocket, path=None):
    # Some websockets versions call the handler with (websocket, path)
    # while others call it with a single connection object. Accept both.
    try:
        conn_path = path if path is not None else getattr(websocket, 'path', None)
    except Exception:
        conn_path = None
    logger.info(f"Client connected: {getattr(websocket, 'remote_address', None)} path={conn_path}")
    try:
        async for msg in websocket:
            if isinstance(msg, (bytes, bytearray)):
                size = len(msg)
                logger.info(f"Received binary chunk: {size} bytes")

                # Send a synthetic asl_batch response
                response = {
                    "type": "asl_batch",
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    "batch_size": 1,
                    "predicted_sign": "A",
                    "confidence": 0.5,
                    "avg_prediction_vector": [0.0]*29,
                }
                await websocket.send(json.dumps(response))
            else:
                # text messages
                logger.info(f"Received text message: {msg}")
                try:
                    data = json.loads(msg)
                    if data.get('type') == 'test':
                        await websocket.send(json.dumps({
                            'type': 'test_response',
                            'message': 'stub running',
                            'timestamp': data.get('timestamp')
                        }))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({'type':'error','message':'invalid json'}))

    except websockets.exceptions.ConnectionClosed:
        logger.info('Client disconnected')

async def main():
    logger.info(f"Starting stub server on ws://{HOST}:{PORT}{PATH}")
    async with websockets.serve(handle_stream, HOST, PORT, max_size=None, compression=None):
        await asyncio.Future()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Server stopped')
