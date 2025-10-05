# Integration Guide: Connect Frontend → ASL Pipeline (stormhacks-2025-api)

This document describes how the current frontend `StreamPreview` component (Next.js) communicates with the API, and gives concrete options for the backend to accept and process the media being sent.

Summary / observed frontend behavior
- The frontend component at `app/stream-preview/stream-preview.tsx` uses the browser MediaRecorder to record the user's camera stream and opens a WebSocket to:
  - `ws://localhost:4000/ws/stream/` (note the `/ws/stream/` path).
- On WebSocket `open` it creates a `MediaRecorder(stream)` and on each `ondataavailable` event sends `event.data` directly over the socket. `event.data` is a Blob (typically a small `video/webm` chunk when using the default codecs).
- The frontend sends raw binary blobs (not JSON). Audio may also be included depending on the MediaRecorder config.

Why this matters
- A raw MediaRecorder blob is not a single image — it is a chunk of encoded media (video/webm by default). The current backend `ASLWebSocketServer` expects JSON `webcam_frame` messages with base64 JPEGs. That contract does not match the live implementation in the web app.

Two practical approaches to make the frontend and backend interoperate

Option A (recommended — minimal backend work): accept per-frame image blobs (JPEG/PNG) over WebSocket
- Change the frontend to capture single frames (canvas.toBlob or ImageCapture) and send each frame as a binary JPEG. This is simple, low-latency, and the backend can decode bytes directly into OpenCV images.
- Advantages:
  - Easy to implement server-side: the server receives JPEG bytes and decodes them with OpenCV or Pillow.
  - Works well for real-time per-frame inference (5–10 fps).
  - Lower CPU/complexity than ingesting a video stream.
- Example server handling (Python, `websockets` library): receive binary message and decode as image

```python
# inside your WebSocket handler (async for message in websocket):
if isinstance(message, (bytes, bytearray)):
    # message contains raw image bytes (e.g. JPEG). Decode with OpenCV
    import numpy as np
    import cv2

    nparr = np.frombuffer(message, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR image
    if img is not None:
        result = self.pipeline.process_frame(img)
        await websocket.send(json.dumps({"type": "asl_prediction", **result}))
    else:
        await websocket.send(json.dumps({"type": "error", "message": "Invalid image bytes"}))
```

Frontend change suggestion (simple): replace MediaRecorder approach with capturing canvas frames and sending blobs
- Use `canvas.getContext('2d').drawImage(video, 0, 0, w, h)` then `canvas.toBlob(blob => ws.send(blob), 'image/jpeg', 0.7)` on a timer.

Option B (keep MediaRecorder video chunks): accept and decode webm streams server-side (heavier)
- The frontend can keep MediaRecorder sending `video/webm` chunks, but the backend must reassemble chunks into a continuous stream and decode frames. This typically requires using ffmpeg to read the webm stream (or saving chunks and extracting frames periodically).
- Trade-offs:
  - More complex on the server. You must buffer webm chunks and feed them to an ffmpeg process that extracts frames for MediaPipe/OpenCV.
  - Potentially lower latency if implemented as a streaming pipeline, but higher CPU and operational complexity.
- Minimal outline (server-side):
  - Accept binary blobs on `/ws/stream/`.
  - Append incoming chunks to a temporary file or pipe them into an ffmpeg subprocess.
  - Use ffmpeg to emit frames (JPEG) which you then decode with OpenCV and feed to the ASL pipeline.

Example ffmpeg (CLI) that reads a webm stream and outputs jpeg frames (for prototyping):

```bash
# Write incoming chunks to a file or a fifo, then run:
ffmpeg -i input.webm -vf fps=5 -q:v 5 frame-%04d.jpg
```

This approach is feasible but more involved; I usually recommend Option A unless your product requires sending continuous encoded video.

Current backend response contract (unchanged)
- When the API returns predictions it sends JSON messages with the following shape (example):

```json
{
  "type": "asl_prediction",
  "timestamp": "2025-10-04T20:00:00Z",
  "hand_detected": true,
  "prediction_vector": [/* 29 floats */],
  "predicted_sign": "A",
  "confidence": 0.92,
  "api_used": "new",
  "error": null
}
```

If an error occurs you'll receive `type: "asl_prediction"` or `type: "error"` with an `error` string.

Recommended immediate server changes (low-risk)
1. Add a dedicated WebSocket path handler for `/ws/stream/` that accepts binary image blobs (JPEG) and decodes them as shown above. Keep the existing JSON-based handler for `webcam_frame` for backwards compatibility.
2. Add simple health endpoint (GET /health) so the frontend can verify reachability before opening WebSocket.
3. If you prefer to keep MediaRecorder video chunks, add a prototype ffmpeg-based pipeline to extract frames from incoming webm chunks.

Security and production notes
- For production use TLS (wss://) and authenticate clients (token in the URL params or an auth message after connect). Do not leave an open unauthenticated inference server on the public internet.
- Consider rate-limiting and per-connection CPU constraints: run a pool of model workers if you expect concurrent users.

Best practices and optimizations
- Prefer sending single-frame JPEG blobs (Option A) at 5–10 fps for a good balance of responsiveness and server load.
- If you really need to stream encoded video (Option B), use a dedicated media server or an ffmpeg subprocess to convert the stream to frames for processing.
- For very low bandwidth, run MediaPipe in the browser and send landmarks (small JSON arrays) instead of images.

Next steps I can implement for you (pick any):
1. Add `/ws/stream/` binary image handler in `asl_pipeline.py` (I can implement the decoding snippet and tests). (low-risk)
2. Add `/health` HTTP endpoint and a small readiness response so the Next.js app can check before connecting. (low-risk)
3. Prototype an ffmpeg-based pipeline to accept `video/webm` chunks from MediaRecorder and extract frames server-side. (medium risk)
4. Create a ready-to-paste Next.js client component that captures canvas frames and sends JPEG blobs (TypeScript/React). I will not modify the web folder unless you ask; I'll provide the drop-in code. (low-risk)

Questions for you
1. Do you want to keep using MediaRecorder and implement server-side webm decoding (Option B), or switch the frontend to send per-frame image blobs (Option A)? I recommend Option A unless you need encoded video on the server.
2. Should I add the `/ws/stream/` binary handler and a `/health` endpoint to the API now? If yes, I'll implement and test them in the API repository.
3. Are there any constraints on CPU/GPU or concurrency (number of simultaneous clients) I should consider when designing the server-side changes?

---

If you'd like, I can implement Option A in `asl_pipeline.py` and provide a small Next.js client snippet you can paste into `stream-preview.tsx` to switch from MediaRecorder to per-frame JPEG blobs. I won't modify any files in `stormhacks-2025-web` unless you ask.


