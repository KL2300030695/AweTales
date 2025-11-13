import asyncio
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from unipipe.audio import TARGET_SR
from unipipe.pipeline_streaming import StreamingDiarizationEngine, StreamingState

app = FastAPI()

sessions: Dict[str, StreamingState] = {}
engine = StreamingDiarizationEngine()


@app.websocket("/ws/session")
async def ws_session(websocket: WebSocket):
    await websocket.accept()
    sid = str(id(websocket))
    sessions[sid] = StreamingState(session_id=sid, sample_rate=TARGET_SR)
    state = sessions[sid]
    try:
        await websocket.send_json({"type": "ready", "session_id": sid, "sr": state.sample_rate})
        while True:
            msg = await websocket.receive_json()
            typ = msg.get("type")
            if typ == "enroll":
                chunk = msg.get("chunk")
                sample_rate = msg.get("sample_rate")
                if not chunk and not msg.get("path"):
                    await websocket.send_json({"type": "error", "message": "missing enrollment data"})
                    continue
                if chunk:
                    await engine.enroll_from_chunk(state, chunk, sample_rate=sample_rate)
                else:
                    await engine.enroll_from_file(state, msg["path"])
                await websocket.send_json({"type": "enrolled", "sr": state.sample_rate})
            elif typ == "audio_chunk":
                if state.target_reference is None:
                    await websocket.send_json({"type": "error", "message": "target speaker not enrolled"})
                    continue
                chunk = msg.get("chunk")
                if not chunk:
                    await websocket.send_json({"type": "error", "message": "chunk missing"})
                    continue
                sample_rate = msg.get("sample_rate", state.sample_rate)
                segments, metadata = await engine.push_chunk(state, chunk, sample_rate=sample_rate)
                if segments:
                    await websocket.send_json(
                        {
                            "type": "partial_result",
                            "segments": [seg.model_dump() for seg in segments],
                            "metadata": metadata,
                        }
                    )
                else:
                    await websocket.send_json({"type": "partial_result", "segments": [], "metadata": {}})
            elif typ == "flush":
                segments, metadata = await engine.flush(state)
                await websocket.send_json(
                    {"type": "final_result", "segments": [seg.model_dump() for seg in segments], "metadata": metadata}
                )
            elif typ == "reset":
                sessions[sid] = StreamingState(session_id=sid, sample_rate=TARGET_SR)
                state = sessions[sid]
                await websocket.send_json({"type": "reset"})
            else:
                await websocket.send_json({"type": "error", "message": f"unknown message type {typ}"})
    except WebSocketDisconnect:
        pass
    finally:
        sessions.pop(sid, None)
