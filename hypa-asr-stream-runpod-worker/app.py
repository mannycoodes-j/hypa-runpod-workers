import logging
import os
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from transcriber import TranscriptionServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
)
logger = logging.getLogger("hypa_asr_stream")

SERVER = TranscriptionServer()
app = FastAPI()


@app.get("/ping")
async def ping() -> dict[str, Any]:
    return {
        "status": "healthy",
        "model": SERVER.model_id,
        "port": os.environ.get("PORT", "80"),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        await SERVER.handle_client(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


@app.websocket("/")
async def websocket_root(websocket: WebSocket) -> None:
    await websocket_endpoint(websocket)
