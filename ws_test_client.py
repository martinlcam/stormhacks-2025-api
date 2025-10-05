import asyncio
import websockets

async def test():
    uri = "ws://localhost:4000/ws/stream/"
    print(f"Connecting to {uri}")
    try:
        async with websockets.connect(uri, max_size=None) as ws:
            # send a small binary payload
            payload = b"\x00\x01\x02\x03" * 1000
            print(f"Sending binary payload of {len(payload)} bytes")
            await ws.send(payload)

            # wait for a response (with timeout)
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                print("Received reply:", resp)
            except asyncio.TimeoutError:
                print("No response within timeout")
    except Exception as e:
        print('Connection error:', e)

if __name__ == '__main__':
    asyncio.run(test())
