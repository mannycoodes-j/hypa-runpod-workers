import asyncio
import base64
import binascii
import io
import json
import logging
import os
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import numpy as np
import torch
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
from transformers import WhisperForConditionalGeneration, WhisperProcessor

log = logging.getLogger("hypa_asr_stream.transcriber")

SAMPLE_RATE = 16_000
DEFAULT_MODEL_ID = "hypaai/Hypa-Whisper-Small-v2"
SILENCE_RMS = 0.01
MIN_AUDIO_SECS = 0.25
MAX_WORKERS = 3

DEFAULT_CONFIG = {
    "temperature": 0.3,
    "beam_size": 1,
    "repetition_penalty": 1.0,
    "top_p": 0.9,
    "no_speech_threshold": 0.6,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_data_uri(value: str) -> str:
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def decode_base64_audio(value: str) -> bytes:
    try:
        return base64.b64decode(_strip_data_uri(value), validate=True)
    except binascii.Error as exc:
        raise ValueError(f"Invalid base64 audio: {exc}") from exc


def decode_wav(raw: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    with io.BytesIO(raw) as buffer, wave.open(buffer, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        pcm = wav_file.readframes(frames)

    if frame_rate != target_sr:
        raise ValueError(
            f"Expected {target_sr} Hz WAV chunks, received {frame_rate} Hz",
        )

    if sample_width == 2:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    elif channels != 1:
        raise ValueError(f"Unsupported channel count: {channels}")

    return audio


class TranscriptionServer:
    def __init__(self) -> None:
        self.model_id = os.getenv("HYPAAI_ASR_MODEL", DEFAULT_MODEL_ID)
        log.info("Loading model from %s", self.model_id)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = WhisperProcessor.from_pretrained(
            self.model_id,
            token=os.getenv("HF_TOKEN") or None,
        )
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=os.getenv("HF_TOKEN") or None,
        )
        self._model.to(self._device)
        if self._device == "cuda":
            self._model = self._model.half()
        self._model.eval()

        worker_count = 1 if self._device == "cuda" else MAX_WORKERS
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="whisper-stream",
        )
        self._connections: set[int] = set()
        self._lock = threading.Lock()
        log.info("Model ready on %s", self._device)

    def _transcribe_hf(self, audio: np.ndarray, config: dict[str, Any]) -> str | None:
        inputs = self._processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self._device)

        temperature = float(config["temperature"])
        generate_kwargs = {
            "num_beams": int(config["beam_size"]),
            "temperature": temperature if temperature > 0 else 1.0,
            "do_sample": temperature > 0,
            "repetition_penalty": float(config["repetition_penalty"]),
            "top_p": float(config["top_p"]) if temperature > 0 else 1.0,
            "task": "transcribe",
        }

        with torch.inference_mode(), torch.autocast(
            self._device,
            enabled=self._device == "cuda",
        ):
            predicted_ids = self._model.generate(input_features, **generate_kwargs)

        text = self._processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )
        result = " ".join(part.strip() for part in text if part.strip())
        return result or None

    async def _send(self, websocket: WebSocket, payload: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(payload))
        except RuntimeError:
            pass

    async def _handle_audio(
        self,
        websocket: WebSocket,
        raw_audio: bytes,
        config: dict[str, Any],
        client_id: int,
    ) -> None:
        if len(raw_audio) < 44:
            return

        try:
            audio = decode_wav(raw_audio)
        except Exception as exc:
            log.warning("Decode failed [%d]: %s", client_id, exc)
            await self._send(websocket, {"type": "decode_error", "message": str(exc)})
            return

        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_AUDIO_SECS:
            return

        if np.sqrt(np.mean(audio**2)) < SILENCE_RMS:
            await self._send(websocket, {"type": "silence", "timestamp": _now()})
            return

        start = perf_counter()
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            self._executor,
            self._transcribe_hf,
            audio,
            config.copy(),
        )
        elapsed = perf_counter() - start

        if text:
            await self._send(
                websocket,
                {
                    "type": "caption",
                    "text": text,
                    "transcript": text,
                    "duration": round(duration, 2),
                    "processing_time": round(elapsed, 3),
                    "rtf": round(elapsed / duration, 3),
                    "timestamp": _now(),
                },
            )
            log.info(
                "[%d] %.1fs audio -> %.2fs | %s",
                client_id,
                duration,
                elapsed,
                text,
            )
        else:
            await self._send(websocket, {"type": "silence", "timestamp": _now()})

    async def handle_client(self, websocket: WebSocket) -> None:
        client_id = id(websocket)
        with self._lock:
            self._connections.add(client_id)

        config = DEFAULT_CONFIG.copy()
        is_busy = False

        await self._send(
            websocket,
            {
                "type": "connected",
                "message": "Transcription server ready",
                "model": self.model_id,
                "config": config,
                "timestamp": _now(),
            },
        )

        try:
            while True:
                message = await websocket.receive()

                if "bytes" in message and message["bytes"] is not None:
                    if is_busy:
                        await self._send(websocket, {"type": "busy", "timestamp": _now()})
                        continue

                    is_busy = True
                    try:
                        await self._handle_audio(
                            websocket,
                            message["bytes"],
                            config,
                            client_id,
                        )
                    finally:
                        is_busy = False
                    continue

                text_message = message.get("text")
                if text_message is None:
                    continue

                try:
                    data = json.loads(text_message)
                except json.JSONDecodeError:
                    continue

                message_type = data.get("type")

                if message_type == "audio_chunk":
                    if is_busy:
                        await self._send(websocket, {"type": "busy", "timestamp": _now()})
                        continue
                    audio_value = data.get("audio")
                    if not isinstance(audio_value, str):
                        await self._send(
                            websocket,
                            {"type": "decode_error", "message": "audio must be base64"},
                        )
                        continue
                    try:
                        raw_audio = decode_base64_audio(audio_value)
                    except ValueError as exc:
                        await self._send(
                            websocket,
                            {"type": "decode_error", "message": str(exc)},
                        )
                        continue

                    is_busy = True
                    try:
                        await self._handle_audio(websocket, raw_audio, config, client_id)
                    finally:
                        is_busy = False

                elif message_type == "config":
                    schema = {
                        "temperature": float,
                        "beam_size": int,
                        "repetition_penalty": float,
                        "top_p": float,
                        "no_speech_threshold": float,
                    }
                    for key, cast in schema.items():
                        if key in data:
                            config[key] = cast(data[key])
                    await self._send(
                        websocket,
                        {"type": "config_updated", "config": config},
                    )

                elif message_type == "ping":
                    await self._send(websocket, {"type": "pong", "timestamp": _now()})

                elif message_type in {"stop", "shutdown"}:
                    await self._send(
                        websocket,
                        {"type": "stopped", "timestamp": _now()},
                    )
                    break

        except WebSocketDisconnect:
            pass
        finally:
            with self._lock:
                self._connections.discard(client_id)
            log.info("Client disconnected [%d]", client_id)
