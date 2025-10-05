// Simple browser client example to send webcam frames to ASL WebSocket server
// Usage: include this script in a client-side page and call `startASLClient()`

async function startASLClient({ wsUrl = 'ws://localhost:4000', fps = 5 } = {}) {
  const video = document.createElement('video');
  video.autoplay = true;
  video.playsInline = true;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
  } catch (err) {
    console.error('Camera access denied or unavailable', err);
    throw err;
  }

  // Create canvas for resizing frames
  const canvas = document.createElement('canvas');
  const targetW = 320;
  const targetH = 240;
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d');

  const ws = new WebSocket(wsUrl);

  ws.addEventListener('open', () => {
    console.log('WebSocket connected to', wsUrl);
  });

  ws.addEventListener('message', (ev) => {
    try {
      const data = JSON.parse(ev.data);
      console.log('Server message:', data);
    } catch (e) {
      console.warn('Received non-JSON message', ev.data);
    }
  });

  let lastSent = 0;

  function sendFrameIfReady() {
    const now = performance.now();
    const minInterval = 1000 / fps;
    if (now - lastSent < minInterval) {
      return;
    }
    lastSent = now;

    // Draw video frame into canvas resized
    ctx.drawImage(video, 0, 0, targetW, targetH);
    // Export JPEG data URL
    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

    const message = {
      type: 'webcam_frame',
      frame_data: dataUrl,
      timestamp: new Date().toISOString(),
    };

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  const interval = setInterval(sendFrameIfReady, 1000 / Math.max(1, fps));

  return {
    stop() {
      clearInterval(interval);
      ws.close();
      const tracks = video.srcObject ? video.srcObject.getTracks() : [];
      tracks.forEach(t => t.stop());
    },
    ws,
    video,
    canvas,
  };
}

// Export for usage in browser module environments
if (typeof window !== 'undefined') {
  window.startASLClient = startASLClient;
}
